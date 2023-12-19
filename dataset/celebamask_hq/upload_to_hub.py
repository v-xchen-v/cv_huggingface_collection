# from datasets import DownloadManager
# from PIL import Image

# dl_manager = DownloadManager()
# extracted_paths = dl_manager.iter_archive('/repos/cv_huggingface_collection/dataset/celebamask_hq/CelebAMask-HQ-label.zip')
# for x in extracted_paths:
#     print(x)
#     path, bufferReader = x
#     Image.open(bufferReader).save('test.png')
# pass


from datasets import load_dataset
username='v-xchen-v'
dataset = load_dataset('/repos/cv_huggingface_collection/dataset/celebamask_hq')
dataset.push_to_hub(f'{username}/celebamask_hq')