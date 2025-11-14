# 🚀 HƯỚNG DẪN TRAINING TỐI ƯU CHO GPU

## ✅ Các thay đổi đã thực hiện:

### 1. **Config tối ưu (config.py)**
```python
MODEL_NAME = "DeepLabV3+"      # Best architecture
ENCODER = "resnet50"             # Good balance
BATCH_SIZE = 16                  # Large batch for GPU
NUM_EPOCHS = 100                 # Enough for convergence
LEARNING_RATE = 3e-4            # Higher LR for faster training
SCHEDULER = "CosineAnnealingWarmRestarts"  # Best scheduler
```

### 2. **Enhanced Data Augmentation**
- Rotation, brightness/contrast
- Hue/saturation/value shifts
- Gaussian blur, motion blur
- Random noise
→ Tăng khả năng generalization

### 3. **Mixed Precision Training**
- Tự động bật khi có GPU
- Tăng tốc ~30-50%
- Giảm VRAM usage ~40%

### 4. **Better Scheduler**
- CosineAnnealingWarmRestarts
- Periodic restarts → thoát local minima
- Better convergence

---

## 📋 BƯỚC THỰC HIỆN TRÊN MÁY MỚI:

### Bước 1: Kiểm tra GPU
```bash
nvidia-smi
```

### Bước 2: Cài đặt PyTorch với CUDA
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (GPU mới hơn)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Bước 3: Cài các dependencies khác
```bash
pip install -r requirements.txt
```

### Bước 4: Kiểm tra setup
```bash
python check_pytorch.py
```
Phải thấy: `CUDA available: True`

### Bước 5: Chạy training
```bash
python train.py
```

### Bước 6: Đánh giá model (sau khi train xong)
```bash
python evaluate.py
```

---

## ⚙️ ĐIỀU CHỈNH THEO GPU:

### GPU 8GB (GTX 1070, RTX 2060)
```python
BATCH_SIZE = 8
IMG_SIZE = (512, 512)
```

### GPU 12GB (RTX 3060, RTX 4060)
```python
BATCH_SIZE = 16  # Đã set sẵn
IMG_SIZE = (512, 512)
```

### GPU 16GB+ (RTX 3080, RTX 4080, A100)
```python
BATCH_SIZE = 24
IMG_SIZE = (640, 640)  # Hoặc giữ 512
```

Nếu bị lỗi CUDA Out of Memory, giảm BATCH_SIZE xuống.

---

## 📊 KỲ VỌNG KẾT QUẢ:

### Với GPU (100 epochs):
- **Training time**: 3-5 giờ (RTX 3060)
- **Epoch time**: ~2-3 phút
- **Expected IoU**: 50-70%
- **Expected Accuracy**: 75-85%

### Theo epoch:
- Epoch 10: IoU ~30-40%
- Epoch 30: IoU ~45-55%
- Epoch 50: IoU ~50-60%
- Epoch 100: IoU ~60-70%

### Model performance:
| IoU | Quality | Description |
|-----|---------|-------------|
| <30% | Poor | Cần train thêm |
| 30-50% | Fair | Đang học |
| 50-70% | Good | Sử dụng được |
| >70% | Excellent | Rất tốt |

---

## 🔍 MONITORING TRAINING:

### Xem logs real-time:
```bash
# Trong khi training, mở terminal khác
tail -f logs/training_*.csv
```

### Checkpoints được lưu tại:
- `checkpoints/best_model.pth` - Model tốt nhất
- `checkpoints/checkpoint_epoch_*.pth` - Mỗi 5 epochs

### Nếu training bị ngắt:
→ Có thể resume từ checkpoint (cần thêm code)

---

## 🎯 SAU KHI TRAINING:

### 1. Evaluate trên test set:
```bash
python evaluate.py
```

### 2. Xem visualizations:
```
checkpoints/visualizations/
├── test_sample_001.jpg
├── test_sample_002.jpg
└── ...
```

### 3. Kiểm tra logs:
```
logs/training_YYYYMMDD_HHMMSS.csv
```

---

## ⚡ TIPS TỐI ƯU:

1. **Dùng NUM_WORKERS > 0** (4-8) để tăng tốc data loading
2. **Pin memory = True** khi dùng GPU
3. **Batch size càng lớn càng tốt** (trong giới hạn VRAM)
4. **Monitor GPU usage**: `watch -n 1 nvidia-smi`
5. **Close các app khác** khi training để giải phóng VRAM

---

## 🐛 XỬ LÝ LỖI:

### CUDA Out of Memory:
```python
BATCH_SIZE = 8  # Giảm xuống
```

### Loss = NaN:
```python
LEARNING_RATE = 1e-4  # Giảm LR
```

### Metrics không tăng:
- Train thêm epochs
- Check data augmentation
- Thử model khác (U-Net vs DeepLabV3+)

---

## 📞 CONTACT/SUPPORT:

Nếu gặp vấn đề:
1. Check logs: `logs/training_*.csv`
2. Check GPU: `nvidia-smi`
3. Test dataset: `python test_dataset.py`
4. Check PyTorch: `python check_pytorch.py`

---

**Good luck with your training! 🚀**
