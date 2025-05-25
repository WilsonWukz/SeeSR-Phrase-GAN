# prompt_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class PromptDataset(Dataset):
    def __init__(self, image_dir, imgid_to_vector, vocab_words, imageid_to_filename):
        self.image_dir = image_dir
        self.imgid_to_vector = imgid_to_vector
        self.vocab_words = vocab_words
        self.imageid_to_filename = imageid_to_filename
        self.image_ids = list(imgid_to_vector.keys())

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  #Align with the CLIP size
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4815, 0.4578, 0.4082),
                                 std=(0.2686, 0.2613, 0.2758)),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_name = self.imageid_to_filename[image_id]  # e.g., "COCO_val2014_000000391895.jpg"
        image_path = os.path.join(self.image_dir, file_name)

        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img)

        label = torch.FloatTensor(self.imgid_to_vector[image_id])  #Multiple labels one-hot
        return img_tensor, label
