import os
import torch

from transformers import MaskFormerForInstanceSegmentation

from huggingface_hub import hf_hub_download
import json

repo_id = f"v-xchen-v/celebamask_hq"
filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
id2label = {int(k):v for k,v in id2label.items()}
print(id2label)

# Replace the head of the pre-trained model
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
checkpoint_path = r'maskformer_celebamaskhq_checkpoint_epoch_0.pth'
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
from torch.utils.data import DataLoader

from transformers import MaskFormerImageProcessor

# Create a preprocessor
preprocessor = MaskFormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
def collate_fn(batch):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    # this function pads the inputs to the same size,
    # and creates a pixel mask
    # actually padding isn't required here since we are cropping
    batch = preprocessor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )

    batch["original_images"] = inputs[2]
    batch["original_segmentation_maps"] = inputs[3]
    
    return batch

import albumentations as A
import numpy as np
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

test_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),

])

# import numpy as np
# from torch.utils.data import Dataset
# import imutils

# class ImageSegmentationDataset(Dataset):
#     """Image segmentation dataset."""

#     def __init__(self, dataset, transform):
#         """
#         Args:
#             dataset
#         """
#         self.dataset = dataset
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, idx):
#         original_image = imutils.resize(np.array(self.dataset[idx]['image']), 512)
#         original_segmentation_map = np.array(self.dataset[idx]['label'])
        
#         transformed = self.transform(image=original_image, mask=original_segmentation_map)
#         image, segmentation_map = transformed['image'], transformed['mask']

#         # convert to C, H, W
#         image = image.transpose(2,0,1)

#         return image, segmentation_map, original_image, original_segmentation_map
     
     
# test_dataset = ImageSegmentationDataset(test_ds, transform=test_transform)
# train_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

