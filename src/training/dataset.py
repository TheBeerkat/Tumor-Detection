from torch.utils.data import Dataset
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np

class INbreastDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []
        self.load_paths()

    def load_paths(self):
        # Load image and mask paths from data/processed
        for file in self.root_dir.glob("*.png"):
            if "_image" in file.name:
                self.image_paths.append(str(file))
            elif "_roi" in file.name:
                self.mask_paths.append(str(file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            image, mask = self.transform(image, mask)
            
        # Convert to float tensors and normalize
        image = torch.from_numpy(image).float() / 255.0
        mask = torch.from_numpy(mask).float() / 255.0
        
        # Add channel dimension
        image = image.unsqueeze(0)  # Shape: (1, H, W)
        mask = mask.unsqueeze(0)    # Shape: (1, H, W)
        
        return image, mask

class Rescale:
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, image, mask):
        return cv2.resize(image, self.output_size), cv2.resize(mask, self.output_size)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    # Path to processed data
    processed_data_path = BASE_DIR / "data/processed"
    
    # Create dataset
    dataset = INbreastDataset(processed_data_path)
    
    # Example usage
    for i in range(5):
        image, mask = dataset[i]
        print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
