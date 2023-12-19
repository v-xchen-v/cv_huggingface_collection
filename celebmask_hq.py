from datasets import load_dataset

all_data = load_dataset('liusq/CelebAMask-HQ', split='train')
print(all_data[0])