"""
PyTorch Dataset cho Food-101N Classification

INPUT (cho model):
    - image_path: Đường dẫn đến ảnh (JPG)
    - Ảnh có thể có kích thước bất kỳ (sẽ được resize về 512x512)

OUTPUT (từ Dataset):
    - image: Tensor (3, 512, 512) - normalized
    - label: Integer (0-100) - class_id
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Food101NDataset(Dataset):
    """
    Dataset cho Food-101N Classification
    
    Hỗ trợ:
        - Load từ train_clean.json hoặc train_all.json
        - Data augmentation với Albumentations
        - Normalization với ImageNet stats
    """
    
    def __init__(self, 
                 data_json_path,
                 img_size=(512, 512),
                 transform=None,
                 is_train=True,
                 use_verified_only=True):
        """
        Args:
            data_json_path: Đường dẫn đến train_*.json hoặc val_*.json
            img_size: Kích thước ảnh (512, 512)
            transform: Albumentations transform (None = dùng default)
            is_train: True = training mode (có augmentation)
            use_verified_only: Chỉ dùng ảnh verified=1 (bỏ qua noisy labels)
        """
        self.img_size = img_size
        self.is_train = is_train
        self.use_verified_only = use_verified_only
        
        # Load data list
        data_json_path = Path(data_json_path)
        with open(data_json_path, 'r', encoding='utf-8') as f:
            self.data_list = json.load(f)
        
        # Filter verified samples if needed
        if use_verified_only and 'verified' in self.data_list[0]:
            original_len = len(self.data_list)
            self.data_list = [item for item in self.data_list if item['verified'] == 1]
            print(f"  Filtered: {len(self.data_list)}/{original_len} verified samples")
        
        print(f"📁 Loaded {len(self.data_list)} samples from {data_json_path.name}")
        
        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
    
    def _get_default_transform(self):
        """Tạo default augmentation"""
        if self.is_train:
            # Training augmentation
            return A.Compose([
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.15, 
                    rotate_limit=15, 
                    p=0.5
                ),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=20,
                        p=1.0
                    ),
                ], p=0.5),
                A.GaussNoise(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # Validation (no augmentation)
            return A.Compose([
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        Trả về 1 sample
        
        Returns:
            image: Tensor (3, 512, 512) - normalized với ImageNet mean/std
            label: Integer - class_id (0-100)
        """
        item = self.data_list[idx]
        
        # Load image
        image_path = item['image_path']
        
        try:
            # Đọc ảnh với PIL (đáng tin cậy hơn)
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)  # Convert to numpy array
        except Exception as e:
            print(f"⚠️  Error loading {image_path}: {e}")
            # Fallback: return black image
            image = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        
        # Apply transform
        transformed = self.transform(image=image)
        image_tensor = transformed['image']  # (3, 512, 512)
        
        # Get label
        label = item['class_id']
        
        return image_tensor, label
    
    def get_class_name(self, class_id, class_map):
        """Helper để lấy class name từ class_id"""
        return class_map['by_id'][str(class_id)]


def create_dataloaders(
    data_dir=None,
    train_json='train_clean.json',
    val_json='val_clean.json',
    batch_size=16,
    num_workers=4,
    img_size=(512, 512),
    use_verified_only=True
):
    """
    Tạo train và validation DataLoaders
    
    Args:
        data_dir: Thư mục chứa data JSON files (None = tự động tìm)
        train_json: Tên file JSON cho train (train_clean.json hoặc train_all.json)
        val_json: Tên file JSON cho val (val_clean.json hoặc val_all.json)
        batch_size: Batch size
        num_workers: Số workers cho DataLoader
        img_size: Kích thước ảnh (512, 512)
        use_verified_only: Chỉ dùng verified samples
    
    Returns:
        train_loader, val_loader, class_map
    """
    # Tự động xác định data_dir nếu không được cung cấp
    if data_dir is None:
        script_dir = Path(__file__).parent  # classification/
        project_root = script_dir.parent  # project/
        data_dir = project_root / 'data' / 'food-101N'
    else:
        data_dir = Path(data_dir)
    
    print("=" * 80)
    print("Creating DataLoaders")
    print("=" * 80)
    
    # Load class map
    with open(data_dir / 'class_map.json', 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    print(f"\n📋 Loaded class_map: {class_map['num_classes']} classes")
    
    # Tạo datasets
    print(f"\n📦 Creating Training Dataset...")
    train_dataset = Food101NDataset(
        data_json_path=data_dir / train_json,
        img_size=img_size,
        is_train=True,
        use_verified_only=use_verified_only
    )
    
    print(f"\n📦 Creating Validation Dataset...")
    val_dataset = Food101NDataset(
        data_json_path=data_dir / val_json,
        img_size=img_size,
        is_train=False,
        use_verified_only=use_verified_only
    )
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✅ DataLoaders created:")
    print(f"  - Train: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    print(f"  - Val:   {len(val_dataset):,} samples, {len(val_loader):,} batches")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image size: {img_size}")
    
    return train_loader, val_loader, class_map


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DEMO: Food-101N Dataset Class")
    print("=" * 80)
    
    # Tạo dataloaders (sử dụng relative paths tự động)
    train_loader, val_loader, class_map = create_dataloaders(
        data_dir=None,  # Tự động tìm
        train_json='train_clean.json',  # Dùng clean data
        val_json='val_clean.json',
        batch_size=8,
        num_workers=2,
        img_size=(512, 512)
    )
    
    # Test 1 batch từ train
    print("\n" + "=" * 80)
    print("TEST: Load 1 batch từ train_loader")
    print("=" * 80)
    
    for images, labels in train_loader:
        print(f"\n📥 INPUT (1 batch from training):")
        print(f"  - images.shape: {images.shape}")
        print(f"    → (batch_size, channels, height, width)")
        print(f"    → ({images.shape[0]}, {images.shape[1]}, {images.shape[2]}, {images.shape[3]})")
        print(f"  - images.dtype: {images.dtype}")
        print(f"  - images.device: {images.device}")
        print(f"  - images value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"    (Normalized với ImageNet mean/std)")
        
        print(f"\n🏷️  LABELS (1 batch):")
        print(f"  - labels.shape: {labels.shape}")
        print(f"  - labels.dtype: {labels.dtype}")
        print(f"  - labels.device: {labels.device}")
        print(f"  - labels values: {labels.tolist()}")
        print(f"  - unique classes in batch: {torch.unique(labels).tolist()}")
        
        # Show class names
        print(f"\n📋 Class names trong batch:")
        for i, label in enumerate(labels[:5]):  # Show first 5
            class_name = class_map['by_id'][str(label.item())]
            print(f"  Sample {i}: Class {label.item():3d} - {class_name}")
        
        break
    
    # Test 1 batch từ val
    print("\n" + "=" * 80)
    print("TEST: Load 1 batch từ val_loader")
    print("=" * 80)
    
    for images, labels in val_loader:
        print(f"\n📥 INPUT (1 batch from validation):")
        print(f"  - images.shape: {images.shape}")
        print(f"  - No augmentation (chỉ resize + normalize)")
        
        print(f"\n🏷️  LABELS:")
        print(f"  - Unique classes: {len(torch.unique(labels))}")
        
        break
    
    print("\n" + "=" * 80)
    print("SUMMARY - Dataset Class")
    print("=" * 80)
    
    print("\n📥 __getitem__ OUTPUT:")
    print("  - image: torch.Tensor")
    print("    + Shape: (3, 512, 512)")
    print("    + Type: torch.float32")
    print("    + Normalized: ImageNet mean=[0.485, 0.456, 0.406]")
    print("                          std=[0.229, 0.224, 0.225]")
    print("    + Value range: approximately [-2.0, 2.5]")
    print("")
    print("  - label: int")
    print("    + Values: 0-100 (101 classes)")
    
    print("\n📦 DataLoader BATCH:")
    print("  - images: torch.Tensor (batch_size, 3, 512, 512)")
    print("  - labels: torch.Tensor (batch_size,)")
    
    print("\n🎨 AUGMENTATION:")
    print("  Training:")
    print("    - Resize to 512x512")
    print("    - Horizontal flip (50%)")
    print("    - Shift/Scale/Rotate")
    print("    - Brightness/Contrast/Hue/Saturation")
    print("    - Gaussian noise")
    print("    - Normalize")
    print("")
    print("  Validation:")
    print("    - Resize to 512x512")
    print("    - Normalize only")
    
    print("\n💡 SỬ DỤNG TRONG TRAINING:")
    print("""
    for images, labels in train_loader:
        images = images.to(device)  # (batch_size, 3, 512, 512)
        labels = labels.to(device)  # (batch_size,)
        
        # Forward
        outputs = model(images)     # (batch_size, 101)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    """)
    
    print("\n" + "=" * 80)
