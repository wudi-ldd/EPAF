# dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

def read_split_files(file_path):
    """Read the file containing the list of image names."""
    with open(file_path, 'r') as f:
        file_names = f.read().strip().split('\n')
    return file_names

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, sam_model, file_list, mask_size=(256, 256), device='cpu'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.sam_model = sam_model
        self.mask_size = mask_size
        self.device = device
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') and f.replace('.png', '') in file_list]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Read image
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # Read mask
        mask_file = image_file.replace('.png', '.png')
        mask_path = os.path.join(self.mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, self.mask_size, interpolation=cv2.INTER_NEAREST)

        # Convert to torch tensor
        input_image_torch = torch.as_tensor(image, dtype=torch.float32)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()  # [C, H, W]

        # Preprocessing step for SAM model
        input_image = self.sam_model.preprocess(input_image_torch.to(self.device))

        # Convert mask to torch tensor
        mask = torch.as_tensor(mask, dtype=torch.long)  # Mask is single-channel

        return input_image, mask
