# Import the necessary packages
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    MaskFormerConfig,
    MaskFormerImageProcessor,
    MaskFormerModel,
    MaskFormerForInstanceSegmentation,
)
import evaluate
from huggingface_hub import notebook_login
import os

# download data from huggingface dataset: https://huggingface.co/datasets/scene_parse_150/viewer/instance_segmentation
# size of downloaded dataset files: ~1.5GB
print("[INFO] Loading the train, val, and test dataset...")
dataset_id = 'scene_parse_150'
subset_name = 'instance_segmentation'
train = load_dataset(dataset_id, name=subset_name, split='train')
validation = load_dataset(dataset_id, name=subset_name, split='validation')
test = load_dataset(dataset_id, name=subset_name, split='test')


# As described in the Scene Parsing dataset page, the instance annotation masks are stored in RGB image format and structured as follows:
# 
# The R(ed) channel encodes category ID.
# The G(reen) channel encodes instance ID.
# Instance IDs are assigned per image such that each object in an annotation image has a different instance ID (regardless of its class ID). In contrast, different annotation images can have objects with the same instance ID. Each image in the dataset has < 256 object instances.

# Grab a random index of the training dataset
print("[INFO] Displaying a random image and its annotation...")
index = random.randint(0, len(train))

# Using the random index grab the corresponding datapoint
# from the training dataset
image = train[index]["image"]
image = np.array(image.convert("RGB"))
annotation = train[index]["annotation"]
annotation = np.array(annotation)

# Plot the original image and the annotations
plt.figure(figsize=(15, 5))
for plot_index in range(3):
    if plot_index == 0:
        # If plot index is 0 display the original image
        plot_image = image
        title = "Original"
    else:
        # Else plot the annotation maps
        plot_image = annotation[..., plot_index - 1]
        title = ["Class Map (R)", "Instance Map (G)"][plot_index - 1]

    # Plot the image
    plt.subplot(1, 3, plot_index + 1)
    plt.imshow(plot_image)
    plt.title(title)
    plt.axis("off")

# Create the MaskFormer Image Preprocessor
preprocessor = MaskFormerImageProcessor(
    reduce_labels=True,
    size=(512, 512),
    ignore_index=255,
    do_resize=False,
    do_rescale=False,
    do_normalize=False,
)

# ## 3) Fine-tuning the MaskFormer Model

# Define the name of the model
model_name = "facebook/maskformer-swin-base-ade"

# Get the MaskFormer config and print it
config = MaskFormerConfig.from_pretrained(model_name)
print("[INFO] displaying the MaskFormer configuration...")
print(config)

# Get a modified version of the id2label and label2id
data = pd.read_csv(
    "./instanceInfo100_train.txt",
    sep="\t",
    header=0,
    on_bad_lines="skip",
)
id2label = {id: label.strip() for id, label in enumerate(data["Object Names"])}
label2id = {v: k for k, v in id2label.items()}

# Edit MaskFormer config labels
config.id2label = id2label
config.label2id = label2id

# Use the config object to initialize a MaskFormer model with randomized weights
model = MaskFormerForInstanceSegmentation(config)

# Replace the randomly initialized model with the pre-trained model weights
base_model = MaskFormerModel.from_pretrained(model_name)
model.model = base_model

# Define the configurations of the transforms specific
# to the dataset used
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# Build the augmentation transforms
train_val_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.HorizontalFlip(p=0.3),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

class ImageSegmentationDataset(Dataset):
    def __init__(self, dataset, processor, transform=None):
        # Initialize the dataset, processor, and transform variables
        self.dataset = dataset
        self.processor = processor
        self.transform = transform
        
    def __len__(self):
        # Return the number of datapoints
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Convert the PIL Image to a NumPy array
        image = np.array(self.dataset[idx]["image"].convert("RGB"))
        
        # Get the pixel wise instance id and category id maps
        # of shape (height, width)
        instance_seg = np.array(self.dataset[idx]["annotation"])[..., 1]
        class_id_map = np.array(self.dataset[idx]["annotation"])[..., 0]
        class_labels = np.unique(class_id_map)

        # Build the instance to class dictionary
        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_seg)
            (image, instance_seg) = (transformed["image"], transformed["mask"])
            
            # Convert from channels last to channels first
            image = image.transpose(2,0,1)

        if class_labels.shape[0] == 1 and class_labels[0] == 0:
            # If the image has no objects then it is skipped
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros(
                (0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1])
            )
        else:
            # Else use process the image with the segmentation maps
            inputs = self.processor(
                [image],
                [instance_seg],
                instance_id_to_semantic_id=inst2class,
                return_tensors="pt"
            )
            inputs = {
                k:v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()
            }

        # Return the inputs
        return inputs

# Build the train and validation instance segmentation dataset
train_dataset = ImageSegmentationDataset(
    train,
    processor=preprocessor,
    transform=train_val_transform
)
val_dataset = ImageSegmentationDataset(
    validation,
    processor=preprocessor,
    transform=train_val_transform
)

# Check if everything is preprocessed correctly
inputs = val_dataset[0]
for k,v in inputs.items():
  print(k, v.shape)

def collate_fn(examples):
    # Get the pixel values, pixel mask, mask labels, and class labels
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_mask = torch.stack([example["pixel_mask"] for example in examples])
    mask_labels = [example["mask_labels"] for example in examples]
    class_labels = [example["class_labels"] for example in examples]

    # Return a dictionary of all the collated features
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels
    }

# Building the training and validation dataloader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Set number of epochs and batch size
num_epochs = 2

# Load checkpoint if it exists
checkpoint_path = r'checkpoint_epoch_1.pth'
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
else: 
    start_epoch = 0

for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch} | Training")

    # Set model in training mode 
    model.train()
    train_loss, val_loss = [], []

    # Training loop
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # Reset the parameter gradients
        optimizer.zero_grad()
 
        # Forward pass
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )

        # Backward propagation
        loss = outputs.loss
        train_loss.append(loss.item())
        loss.backward()

        if idx % 50 == 0:
            print("  Training loss: ", round(sum(train_loss)/len(train_loss), 6))

        # Optimization
        optimizer.step()
        
        del loss
        torch.cuda.empty_cache()

    # Average train epoch loss
    train_loss = sum(train_loss)/len(train_loss)

    # Set model in evaluation mode
    model.eval()

    start_idx = 0
    print(f"Epoch {epoch} | Validation")
    for idx, batch in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            # Get validation loss
            loss = outputs.loss
            val_loss.append(loss.item())

            if idx % 50 == 0:
                print("  Validation loss: ", round(sum(val_loss)/len(val_loss), 6))

    # Average validation epoch loss
    val_loss = sum(val_loss)/len(val_loss)

    # Saving checkpoint here
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
    
    # Print epoch losses
    print(f"Epoch {epoch} | train_loss: {train_loss} | validation_loss: {val_loss}")

# ## 4) Evaluating the MaskFormer Model
# We won't be using albumentations to preprocess images for inference
preprocessor.do_normalize = True
preprocessor.do_resize = True
preprocessor.do_rescale = True

# Use random test image
index = random.randint(0, len(test))
image = test[index]["image"].convert("RGB")
target_size = image.size[::-1]

# Preprocess image
inputs = preprocessor(images=image, return_tensors="pt").to(device)

# Inference
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

# Let's print the items returned by our model and their shapes
print("Outputs...")
for key, value in outputs.items():
    print(f"  {key}: {value.shape}")
    
    # Post-process results to retrieve instance segmentation maps
result = preprocessor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    target_sizes=[target_size]
)[0] # we pass a single output therefore we take the first result (single)

instance_seg_mask = result["segmentation"].cpu().detach().numpy()

print(f"Final mask shape: {instance_seg_mask.shape}")
print("Segments Information...")
for info in result["segments_info"]:
    print(f"  {info}")
    
def visualize_instance_seg_mask(mask):
    # Initialize image with zeros with the image resolution
    # of the segmentation mask and 3 channels
    image = np.zeros((mask.shape[0], mask.shape[1], 3))

    # Create labels
    labels = np.unique(mask)
    label2color = {
        label: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for label in labels
    }

    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            image[height, width, :] = label2color[mask[height, width]]

    image = image / 255
    return image

instance_seg_mask_disp = visualize_instance_seg_mask(instance_seg_mask)

plt.figure(figsize=(10, 10))
for plot_index in range(2):
    if plot_index == 0:
        plot_image = image
        title = "Original"
    else:
        plot_image = instance_seg_mask_disp
        title = "Segmentation"
    
    plt.subplot(1, 2, plot_index+1)
    plt.imshow(plot_image)
    plt.title(title)
    plt.axis("off")
    plt.savefig('test.png')   
# # Load Mean IoU metric
# metrics = evaluate.load("mean_iou")

# # Set model in evaluation mode
# model.eval()

# # Test set doesn't have annotations so we will use the validation set
# ground_truths, preds = [], []

# for idx in tqdm(range(200)):
#     image = validation[idx]["image"].convert("RGB")
#     target_size = image.size[::-1]

#     # Get ground truth semantic segmentation map
#     annotation = np.array(validation[idx]["annotation"])[:,:,0]
#     # Replace null class (0) with the ignore_index (255) and reduce labels
#     annotation -= 1
#     annotation[annotation==-1] = 255
#     ground_truths.append(annotation)

#     # Preprocess image
#     inputs = preprocessor(images=image, return_tensors="pt").to(device)

#     # Inference
#     model.eval()
#     with torch.no_grad():
#         outputs = model(**inputs)
 
#     # Post-process results to retrieve semantic segmentation maps
#     result = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=[target_size])[0]
#     semantic_seg_mask = result.cpu().detach().numpy()
#     preds.append(semantic_seg_mask)

# results = metrics.compute(
#     predictions=preds,
#     references=ground_truths,
#     num_labels=100,
#     ignore_index=255
# )
# print(f"Mean IoU: {results['mean_iou']} | Mean Accuracy: {results['mean_accuracy']} | Overall Accuracy: {results['overall_accuracy']}")