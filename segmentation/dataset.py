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
        if image is None:
            raise ValueError(f"Cannot read image: {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask (single channel, values 0..103)
        mask = cv2.imread(sample['mask'], cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read mask: {sample['mask']}")
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # If no transform, convert to tensor manually
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Ensure mask is tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        
        # Convert mask to long tensor for CrossEntropyLoss
        mask = mask.long()
        
        # Don't return unique_labels as it has variable length and breaks batching
        return image, mask
    
    def get_class_weights(self):
        """
        Calculate class weights for handling class imbalance.
        Weights are inversely proportional to class frequency.
        
        Returns:
            numpy array of shape (num_classes,) with weights for each class
        """
        print("Calculating class weights from training data...")
        
        # Count pixels for each class
        class_counts = np.zeros(self.num_classes, dtype=np.int64)
        
        from tqdm import tqdm
        for sample in tqdm(self.samples, desc="Computing class weights"):
            mask = cv2.imread(sample['mask'], cv2.IMREAD_UNCHANGED)
            
            # Count occurrences of each class
            unique, counts = np.unique(mask, return_counts=True)
            for class_id, count in zip(unique, counts):
                if class_id < self.num_classes:
                    class_counts[class_id] += count
        
        # Calculate weights: inverse frequency
        # Add small epsilon to avoid division by zero
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (class_counts + 1e-6)
        
        # Normalize weights so that mean = 1.0
        class_weights = class_weights / class_weights.mean()
        
        # Cap maximum weight to avoid extreme values
        class_weights = np.clip(class_weights, 0.1, 10.0)
        
        print(f"Class weights computed:")
        print(f"  - Min weight: {class_weights.min():.4f}")
        print(f"  - Max weight: {class_weights.max():.4f}")
        print(f"  - Mean weight: {class_weights.mean():.4f}")
        
        return class_weights


def get_train_transform(img_size=(512, 512)):
    """Training augmentations - Enhanced for better generalization"""
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        # Color/intensity transforms
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        # Blur/noise for robustness
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        # Normalization (ImageNet stats)
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
