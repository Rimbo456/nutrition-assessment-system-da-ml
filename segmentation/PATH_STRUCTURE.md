# 📁 Cấu trúc thư mục dự án

## Sau khi thay đổi sang đường dẫn tương đối

Tất cả các đường dẫn tuyệt đối đã được thay đổi thành đường dẫn tương đối.
Code giờ có thể chạy trên **bất kỳ máy nào** mà không cần thay đổi đường dẫn!

---

## 📂 Cấu trúc thư mục yêu cầu:

```
KLTN/
├── foodseg103/                    # Dataset gốc
│   ├── train/
│   │   ├── img/
│   │   └── ann/
│   ├── test/
│   │   ├── img/
│   │   └── ann/
│   └── meta.json
│
└── project/                       # Code của bạn
    ├── segmentation/              # Thư mục chính
    │   ├── config.py              # Config chính
    │   ├── train.py               # Training script
    │   ├── dataset.py             # Dataset loader
    │   ├── evaluate.py            # Evaluation script
    │   └── ...
    │
    ├── data/                      # Dữ liệu đã preprocess
    │   └── foodseg103/
    │       ├── images/
    │       │   ├── train/
    │       │   ├── val/
    │       │   └── test/
    │       ├── annotations/
    │       │   ├── train/
    │       │   ├── val/
    │       │   └── test/
    │       ├── manifest_train.csv
    │       ├── manifest_val.csv
    │       ├── manifest_test.csv
    │       └── class_map.json
    │
    ├── checkpoints/               # Model checkpoints (tự tạo)
    │   └── best_model.pth
    │
    └── logs/                      # Training logs (tự tạo)
        └── training_*.csv
```

---

## ✅ Kiểm tra đường dẫn

Chạy script test để kiểm tra:

```bash
cd project/segmentation
python test_paths.py
```

Nếu thấy:
```
✅ All data paths are valid!
✅ Ready for training!
```
→ Mọi thứ đã OK!

---

## 🚀 Chạy trên máy khác

### Bước 1: Clone/Copy code
```bash
git clone <repo_url>
cd KLTN/project/segmentation
```

### Bước 2: Kiểm tra cấu trúc
```bash
python test_paths.py
```

### Bước 3: Nếu chưa có dữ liệu, preprocess
```bash
python preprocess_foodseg103.py
```

### Bước 4: Train!
```bash
python train.py
```

---

## 📝 Lưu ý

- **Tất cả đường dẫn tự động tính toán** dựa trên vị trí file
- **Không cần thay đổi config** khi chuyển máy
- **Cấu trúc thư mục phải đúng** như trên

---

## 🔧 Files đã cập nhật

- ✅ `config.py` - Đường dẫn tương đối
- ✅ `config_cpu.py` - Đường dẫn tương đối  
- ✅ `config_gpu.py` - Đường dẫn tương đối
- ✅ `inference_demo.py` - Đường dẫn tương đối
- ✅ `preprocess_foodseg103.py` - Đường dẫn tương đối
- ✅ `validate_masks.py` - Đường dẫn tương đối
- ✅ `decode_one_ann.py` - Đường dẫn tương đối
- ✅ `inspect_mask_one.py` - Đường dẫn tương đối

---

**Code giờ đã portable! 🎉**
