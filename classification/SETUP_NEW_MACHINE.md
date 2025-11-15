# Hướng dẫn chuyển sang máy khác để training

## 🚀 Bước 1: Chuyển đổi paths (chỉ cần làm 1 lần)

Nếu các file JSON trong `data/food-101N/` đang chứa absolute paths (đường dẫn tuyệt đối), chạy:

```bash
cd project/classification
python convert_paths.py
```

Script này sẽ:
- Convert tất cả absolute paths → relative paths
- Update các file: `train_all.json`, `train_clean.json`, `val_all.json`, `val_clean.json`

**Ví dụ conversion:**
```
TRƯỚC: "d:\\Dev\\University\\KLTN\\food-101N\\images\\apple_pie\\xxx.jpg"
SAU:   "..\\..\\..\\food-101N\\images\\apple_pie\\xxx.jpg"
```

## 📦 Bước 2: Copy sang máy mới

Copy toàn bộ thư mục `KLTN/` sang máy mới, **giữ nguyên cấu trúc**:

```
KLTN/
├── food-101N/              # Dataset gốc (images + meta)
└── project/
    ├── data/
    │   └── food-101N/      # JSON files (đã có relative paths)
    └── classification/     # Scripts
```

## 💻 Bước 3: Training trên máy mới

```bash
# 1. Di chuyển vào thư mục classification
cd KLTN/project/classification

# 2. (Optional) Kiểm tra dataset hoạt động
python dataset_food101n.py

# 3. Training
python train.py
```

## ✅ Checklist

- [ ] Đã chạy `convert_paths.py` để convert sang relative paths
- [ ] Đã copy toàn bộ thư mục `KLTN/` (bao gồm cả `food-101N/`)
- [ ] Cấu trúc thư mục giữ nguyên như trên
- [ ] Máy mới có GPU (recommended) hoặc CPU
- [ ] Đã cài đặt dependencies:
  ```bash
  pip install torch torchvision
  pip install opencv-python pillow
  pip install albumentations tqdm numpy
  ```

## 🔧 Troubleshooting

### Lỗi: "Cannot read image"

**Nguyên nhân**: Paths chưa được convert hoặc cấu trúc thư mục sai

**Giải pháp**:
1. Kiểm tra file JSON: `cat data/food-101N/train_clean.json | head -20`
2. Đảm bảo paths là relative (bắt đầu với `../` hoặc `..\\`)
3. Nếu vẫn là absolute, chạy lại `python convert_paths.py`

### Lỗi: "File not found"

**Nguyên nhân**: Thiếu thư mục `food-101N/`

**Giải pháp**: 
- Đảm bảo copy cả thư mục `food-101N/` (dataset gốc)
- Cấu trúc phải là: `KLTN/food-101N/images/...`

### Kiểm tra paths đang dùng

```python
import json
from pathlib import Path

# Kiểm tra JSON file
data_dir = Path("../data/food-101N")
with open(data_dir / "train_clean.json") as f:
    data = json.load(f)
    
print("Sample paths:")
for i in range(3):
    print(f"  {data[i]['image_path']}")
    
# Kiểm tra absolute/relative
first_path = Path(data[0]['image_path'])
print(f"\nIs absolute: {first_path.is_absolute()}")
```

## 📝 Notes

- **Relative paths** giúp code portable (dễ chuyển máy)
- **Không** cần chạy lại `preprocess_food101n.py` trên máy mới
- Dataset gốc (`food-101N/`) chỉ cần copy 1 lần
- JSON files đã có sẵn, chỉ cần đảm bảo paths đúng

---

**Date**: November 2025
**Version**: 2.0 (with relative paths support)
