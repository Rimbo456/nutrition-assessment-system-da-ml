# Food-101N Classification

Pipeline tiền xử lý và training cho bài toán **Food Classification** với dataset **Food-101N**.

## 📚 Tổng quan

- **Dataset**: Food-101N (101 loại thực phẩm)
- **Task**: Image Classification
- **INPUT**: Ảnh JPG kích thước bất kỳ → resize về **512x512**
- **OUTPUT**: Class ID (0-100) và confidence scores

### Đặc điểm Food-101N

- **101 classes** thực phẩm
- **~310,000 ảnh** với **noisy labels** (có nhiễu)
- **Verification labels**: Đánh dấu ảnh đúng (verified=1) hoặc sai (verified=0)
- Dataset được chia sẵn thành **train** và **validation**

---

## 🗂️ Cấu trúc thư mục

```
KLTN/
├── food-101N/                          # Dataset gốc (đã giải nén)
│   ├── images/
│   │   ├── apple_pie/
│   │   ├── baby_back_ribs/
│   │   └── ... (101 classes)
│   └── meta/
│       ├── classes.txt
│       ├── verified_train.tsv
│       └── verified_val.tsv
│
└── project/
    ├── data/
    │   └── food-101N/                  # Dữ liệu đã preprocess
    │       ├── class_map.json
    │       ├── dataset_stats.json
    │       ├── train_clean.json        # Train (verified=1)
    │       ├── train_all.json          # Train (all + verified labels)
    │       ├── val_clean.json          # Val (verified=1)
    │       └── val_all.json            # Val (all + verified labels)
    │
    └── classification/                 # Scripts
        ├── preprocess_food101n.py      # Script tiền xử lý
        ├── dataset_food101n.py         # PyTorch Dataset
        ├── config.py                   # Configuration
        ├── README.md                   # File này
        ├── checkpoints/                # Model checkpoints (sẽ tạo)
        ├── logs/                       # Training logs (sẽ tạo)
        └── results/                    # Results (sẽ tạo)
```

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Cài đặt dependencies

```bash
# Di chuyển vào thư mục classification
cd project/classification

pip install torch torchvision
pip install opencv-python pillow
pip install albumentations
pip install tqdm numpy
```

### Bước 2: Preprocessing Dataset

Chạy script tiền xử lý (tự động tìm đường dẫn):

```bash
# Đảm bảo bạn đang ở thư mục project/classification
python preprocess_food101n.py
```

**Lưu ý**: Script sẽ tự động tìm:
- Dataset gốc: `../../food-101N/`
- Output: `../data/food-101N/`

**Script sẽ:**
1. ✅ Load 101 classes từ `meta/classes.txt`
2. ✅ Load train/val data với verification labels
3. ✅ Tạo clean splits (chỉ verified=1)
4. ✅ Phân tích thống kê dataset
5. ✅ Verify images (sample check)
6. ✅ Lưu metadata JSON files
7. ✅ Phân tích kích thước ảnh

**Output:**
```
project/data/food-101N/
├── class_map.json          # 101 classes mapping
├── dataset_stats.json      # Thống kê dataset
├── image_stats.json        # Thống kê kích thước ảnh
├── train_clean.json        # ~200k+ samples (verified=1)
├── train_all.json          # ~250k+ samples (all)
├── val_clean.json          # ~50k+ samples (verified=1)
└── val_all.json            # ~60k+ samples (all)
```

### Bước 3: Test Dataset Class

Kiểm tra Dataset có hoạt động đúng không:

```bash
python dataset_food101n.py
```

**Expected output:**
```
📥 INPUT (1 batch from training):
  - images.shape: torch.Size([8, 3, 512, 512])
  - images.dtype: torch.float32
  - images value range: [-2.118, 2.640]

🏷️ LABELS (1 batch):
  - labels.shape: torch.Size([8])
  - labels values: [45, 12, 89, 3, 67, 23, 91, 56]
```

### Bước 4: Training (Coming soon)

Tạo file `train.py` để training model.

---

## 📊 Dataset Information

### Food-101N Statistics

Sau khi chạy preprocessing, bạn sẽ có:

```
📊 Dataset Statistics:
  Total classes: 101
  Total images: ~310,000

  Training set:
    - Total: ~250,000
    - Clean: ~200,000 (80%)
    - Noisy: ~50,000 (20%)

  Validation set:
    - Total: ~60,000
    - Clean: ~50,000 (83%)
    - Noisy: ~10,000 (17%)
```

### Input/Output Specification

#### INPUT (cho model):
```python
image: torch.Tensor
    - Shape: (batch_size, 3, 512, 512)
    - Type: torch.float32
    - Normalized với ImageNet stats:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    - Value range: approximately [-2.5, 2.5]
```

#### OUTPUT (từ model):
```python
logits: torch.Tensor
    - Shape: (batch_size, 101)
    - Type: torch.float32
    - Raw scores (chưa softmax)

prediction: int (sau argmax)
    - Values: 0-100 (class_id)
```

---

## 🎨 Data Augmentation

### Training Augmentation
```python
- Resize to 512x512
- Horizontal flip (50% probability)
- Shift/Scale/Rotate (±10%, ±15%, ±15°)
- Random brightness/contrast adjustment
- Random hue/saturation/value adjustment
- Gaussian noise (20% probability)
- Normalize (ImageNet mean/std)
```

### Validation
```python
- Resize to 512x512
- Normalize only (no augmentation)
```

---

## ⚙️ Configuration

Chỉnh sửa `config.py` để thay đổi settings:

```python
# Model
MODEL_NAME = "resnet50"    # resnet50, efficientnet_b3, vit_b_16
NUM_CLASSES = 101
PRETRAINED = True

# Training
BATCH_SIZE = 16            # Giảm xuống 8 nếu GPU memory không đủ
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# Data
TRAIN_JSON = 'train_clean.json'  # Hoặc 'train_all.json'
VAL_JSON = 'val_clean.json'      # Hoặc 'val_all.json'
USE_VERIFIED_ONLY = True         # Filter verified samples

# Image
IMG_SIZE = (512, 512)
```

---

## 💡 Best Practices

### 1. Dealing with Noisy Labels

Food-101N có **noisy labels**, bạn có 3 options:

**Option 1: Dùng Clean Data (Recommended)**
```python
# Trong config.py
TRAIN_JSON = 'train_clean.json'  # Chỉ dùng verified=1
VAL_JSON = 'val_clean.json'
```

**Option 2: Dùng All Data + Filter trong Dataset**
```python
# Trong config.py
TRAIN_JSON = 'train_all.json'
USE_VERIFIED_ONLY = True  # Filter trong Dataset class
```

**Option 3: Dùng All Data + Noise Handling**
```python
TRAIN_JSON = 'train_all.json'
USE_VERIFIED_ONLY = False

# Áp dụng techniques:
# - Label Smoothing
# - Focal Loss
# - Mixup/CutMix augmentation
# - Bootstrap/Co-teaching
```

### 2. Training Tips

- ✅ Sử dụng **pretrained weights** (ImageNet)
- ✅ Áp dụng **strong augmentation** (rotation, color jitter, noise)
- ✅ Sử dụng **mixed precision** (AMP) để tăng tốc
- ✅ Monitor **validation metrics** để tránh overfitting
- ✅ Sử dụng **learning rate scheduler** (ReduceLROnPlateau)
- ✅ Áp dụng **early stopping**

### 3. Performance Optimization

```python
# Tăng tốc DataLoader
NUM_WORKERS = 8        # Tăng nếu CPU mạnh
PIN_MEMORY = True      # Pin memory cho GPU
PREFETCH_FACTOR = 2    # Prefetch batches

# Giảm memory usage
BATCH_SIZE = 8         # Giảm nếu OOM
USE_AMP = True         # Mixed precision
```

---

## 📝 Code Examples

### Example 1: Load và visualize 1 sample

```python
from dataset_food101n import Food101NDataset
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import json

# Tự động tìm đường dẫn
script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data' / 'food-101N'

# Load dataset
dataset = Food101NDataset(
    data_json_path=data_dir / 'train_clean.json',
    is_train=False  # No augmentation
)

# Load class map
with open(data_dir / 'class_map.json', 'r') as f:
    class_map = json.load(f)

# Get 1 sample
image, label = dataset[0]

# Denormalize để visualize
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
image_denorm = image * std + mean

# Plot
plt.imshow(image_denorm.permute(1, 2, 0).clip(0, 1))
plt.title(f"Class {label}: {class_map['by_id'][str(label)]}")
plt.axis('off')
plt.show()
```

### Example 2: Tạo DataLoaders

```python
from dataset_food101n import create_dataloaders

# Tạo dataloaders (tự động tìm đường dẫn)
train_loader, val_loader, class_map = create_dataloaders(
    data_dir=None,  # Tự động: ../data/food-101N
    train_json='train_clean.json',
    val_json='val_clean.json',
    batch_size=16,
    num_workers=4
)

# Iterate
for images, labels in train_loader:
    print(f"Batch: {images.shape}")
    # Process batch...
    break
```

### Example 3: Training loop (skeleton)

```python
import torch
import torch.nn as nn
from torchvision import models
from dataset_food101n import create_dataloaders
import config

# Setup
device = config.DEVICE

# Load data
train_loader, val_loader, class_map = create_dataloaders(
    data_dir=config.DATA_ROOT,
    train_json=config.TRAIN_JSON,
    val_json=config.VAL_JSON,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS
)

# Create model
model = models.resnet50(pretrained=config.PRETRAINED)
model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
model = model.to(device)

# Optimizer & loss
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY
)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(config.NUM_EPOCHS):
    model.train()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % config.LOG_INTERVAL == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Validation
    # ... (implement validation)
```

---

## 🐛 Troubleshooting

### Lỗi: "Cannot read image"
```
⚠️ Giải pháp:
- Kiểm tra đường dẫn trong JSON files
- Verify dataset đã giải nén đúng
- Chạy lại preprocessing
```

### Lỗi: CUDA out of memory
```
⚠️ Giải pháp:
- Giảm BATCH_SIZE xuống 8 hoặc 4
- Giảm IMG_SIZE xuống (256, 256)
- Enable gradient accumulation
- Sử dụng mixed precision (USE_AMP = True)
```

### Dataset load chậm
```
⚠️ Giải pháp:
- Tăng NUM_WORKERS lên 8
- Sử dụng SSD thay vì HDD
- Giảm augmentation complexity
```

### Lỗi: ModuleNotFoundError
```bash
# Cài đặt dependencies
pip install albumentations
pip install opencv-python
pip install pillow
```

---

## 📚 References

- [Food-101N Paper: CleanNet](https://kuanghuei.github.io/Food-101N/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Albumentations](https://albumentations.ai/docs/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)

---

## 📞 Next Steps

1. ✅ Preprocessing completed
2. ✅ Dataset class created
3. ✅ Config setup
4. ⏳ Create `train.py` for training
5. ⏳ Create `evaluate.py` for evaluation
6. ⏳ Create `inference.py` for prediction

---

## 📝 Notes

- Dataset gốc nằm trong `KLTN/food-101N/` (READ-ONLY)
- Dữ liệu processed nằm trong `KLTN/project/data/food-101N/`
- Scripts nằm trong `KLTN/project/classification/`
- **KHÔNG** copy ảnh - chỉ lưu paths trong JSON
- Preprocessing on-the-fly trong Dataset class
- **Tất cả paths sử dụng relative paths** - dễ dàng chuyển máy

---

**Author**: AI Assistant  
**Date**: November 2025  
**Version**: 1.0
