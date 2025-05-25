# prompt_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class PromptDataset(Dataset):
    def __init__(self, image_dir, imgid_to_vector, vocab_words, imageid_to_filename):
        """
        COCO 格式兼容的 Prompt-GAN Dataset 构造器

        Args:
            image_dir (str): 图像文件夹路径（如 val2014）
            imgid_to_vector (dict): image_id → 标签向量
            vocab_words (list): 所有形容词词表
            imageid_to_filename (dict): image_id → 图像文件名（如 COCO_val2014_000000123456.jpg）
        """
        self.image_dir = image_dir
        self.imgid_to_vector = imgid_to_vector
        self.vocab_words = vocab_words
        self.imageid_to_filename = imageid_to_filename
        self.image_ids = list(imgid_to_vector.keys())

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 与 CLIP 尺寸对齐
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

        label = torch.FloatTensor(self.imgid_to_vector[image_id])  # 多标签 one-hot
        return img_tensor, label