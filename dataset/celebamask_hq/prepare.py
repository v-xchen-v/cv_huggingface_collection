import os
from imutils import paths
import random
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import json

# the larger id could cover the smaller id in label image, so that the label order is meaningful
labels = [\
    'hair',
    'hat',
    'ear_r', # earing, earing id is smaller than skin, means skin will cover earing in label
    'skin',
    'l_brow', # left eyebrow, eyebrow id is larger means eyebrow will prodiect even if covered by hair/hat
    'r_brow', # right eyebrow
    'l_eye', # left eye
    'r_eye', # right eye
    'eye_g', # eyeglass
    'nose',
    'u_lip', # upperlip
    'l_lip', # lowerlip
    'mouth',
    'cloth', 
    'neck',
    'neck_l', # necklace
    'l_ear', # left ear
    'r_ear', # right ear
]

# +1 is for reserved background class
label2id = {label: i+1 for i, label in enumerate(labels)}
id2label = {i+1: label for i, label in enumerate(labels)}

def dump_json_files():
    # Saving the dictionary to a JSON file
    with open('dataset/celebamask_hq/label2id.json', 'w') as file:
        json.dump(label2id, file, indent=4)

    with open('dataset/celebamask_hq/id2label.json', 'w') as file:
        json.dump(id2label, file, indent=4)

num_classes = len(labels)
        
dataset_dir = '/cv_demo_fun_data/CelebAMask-HQ'
images_dir = Path(dataset_dir) / 'CelebA-HQ-img'
annotations_dir = Path(dataset_dir) / 'CelebAMask-HQ-mask-anno'
images_path = [x for x in paths.list_images(images_dir)]
annotations_path = [x for x in paths.list_images(annotations_dir)]

def get_id(image_path):
    id = image_path.split(os.path.sep)[-1].split('.')[0]
    return id

def find_annotation_path(id):
    paths = []
    for path in annotations_path:
        filename = path.split(os.path.sep)[-1].split('.')[0]
        if filename.split('_')[0] == id.zfill(5): # fill with leading zeros if necessary
            paths.append(path)
    return paths

def get_anno_attribute(anno_image_path):
    filename = anno_image_path.split(os.path.sep)[-1].split('.')[0]
    return '_'.join(filename.split('_')[1:])
    # return anno_image_path.split(os.path.sep)[-1].split('.')[0].replace(f'{id}_', '')
    
def list_anno_attributes(id):
    paths = find_annotation_path(id)
    return [get_anno_attribute(path) for path in paths]

def color_palette():
    """Color palette that maps each class to RGB values.
    
    This one is actually taken from ADE20k.
    """
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

palette = color_palette()

import numpy as np

for idx, image_path in enumerate(tqdm(images_path)):
    image = Image.open(image_path).resize((512, 512))
    id = get_id(image_path)
    annotation_images_path = find_annotation_path(id)

    # create a single channel image with size of 512x512
    segmentation_map = np.zeros((512, 512), dtype=np.uint8)
    annotation_images_path.sort(key=lambda x: label2id.get(get_anno_attribute(x)))
    for path in annotation_images_path:
        attribute = get_anno_attribute(path)
        label_id = label2id[attribute]

        # open as single channel image
        anno = Image.open(path).convert('L')
        # set the non-zero values to 1
        anno = np.array(anno)
        segmentation_map[anno > 0] = label_id
        Image.fromarray(segmentation_map).save(f'dataset/celebamask_hq/label/{id}_label.png')
        
    if idx < 20:
        color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
        for label, color in enumerate(palette):
            color_segmentation_map[segmentation_map - 1 == label, :] = color
        # Convert to BGR
        ground_truth_color_seg = color_segmentation_map[..., ::-1]

        img = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
        img = img.astype(np.uint8)

        Image.fromarray(img).save(f'dataset/celebamask_hq/label_vis/{id}_labeledimg.png')
        Image.fromarray(segmentation_map*10).save(f'dataset/celebamask_hq/label_vis/{id}_label.png')
