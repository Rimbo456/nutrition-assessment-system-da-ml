"""
PyTorch Dataset for FoodSeg103 preprocessed data.
Reads from manifest CSV files.
"""
import csv
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FoodSeg103Dataset(Dataset):
    def __init__(self, manifest_path, transform=None, num_classes=104):
        """
        Args:
            manifest_path: path to manifest_{split}.csv
            transform: albumentations transform
            num_classes: 104 (0=background + 103 food classes)
        """
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.num_classes = num_classes
        self.samples = []
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    'image': row['image_path'],
                    'mask': row['mask_path']
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Read image (BGR -> RGB)
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask (single channel, values 0..103)
        mask = cv2.imread(sample['mask'], cv2.IMREAD_UNCHANGED)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to long tensor for CrossEntropyLoss
        mask = torch.as_tensor(mask, dtype=torch.long)
        
        return image, mask


def get_train_transform(img_size=(512, 512)):
    """Training augmentations"""
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transform(img_size=(512, 512)):
    """Validation transforms (no augmentation)"""
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
