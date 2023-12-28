import datasets
import os
from PIL import Image
import imutils

_CITATION = """\
@inproceedings{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
"""
   
class CelebAMaskHQ(datasets.GeneratorBasedBuilder):
# class CelebAMaskHQ:
    """CelebAMaskHQ dataset."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labels = [\
            'hair',
            'hat',
            'skin',
            'l_brow', # left eyebrow
            'r_brow', # right eyebrow
            'l_ear', # left ear
            'r_ear', # right ear
            'ear_r', # earing
            'l_eye', # left eye
            'r_eye', # right eye
            'eye_g', # eyeglass
            'nose',
            'u_lip' # upperlip
            'l_lip', # lowerlip
            'mouth',
            'neck',
            'neck_l', # necklace
            'cloth', 
        ]
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id['background'] = 0
        self.id2label[0]='background' 
        self.num_classes = len(self.labels)
        
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "image": datasets.Image(),
                "label": datasets.Image(),
            }),
            supervised_keys=None,
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={
                    "images": dl_manager.iter_archive('/repos/cv_huggingface_collection/dataset/celebamask_hq/CelebAMask-HQ-img.zip'),
                    "labels": dl_manager.extract('/repos/cv_huggingface_collection/dataset/celebamask_hq/CelebAMask-HQ-label.zip')
                })
        ]
        
    def _generate_examples(self, **kwargs):
        images_iter = kwargs["images"]
        labels_root = kwargs["labels"]

        def get_id(image_path):
            id = image_path.split(os.path.sep)[-1].split('.')[0]
            return id
        
        for image_path, image_bufferReader in images_iter:
            id = get_id(image_path)
            label_relapath = image_path.replace('CelebAMask-HQ-img', 'CelebAMask-HQ-label').replace(f'{id}.jpg', f'{id}_label.png')
            label_pathfile = os.path.join(labels_root, label_relapath)
            yield get_id(image_path), {
                "image": Image.open(image_bufferReader),
                "label": Image.open(label_pathfile).convert('L'),
            }