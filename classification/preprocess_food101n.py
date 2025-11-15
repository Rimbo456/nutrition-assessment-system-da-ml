"""
Script tiền xử lý dữ liệu Food-101N cho mô hình Classification

Dataset Food-101N:
- 101 classes thực phẩm
- ~310,000 ảnh với noisy labels
- Có verification labels cho train/val để xác định ảnh đúng/sai

INPUT:
    - Ảnh JPG từ food-101N/images/
    - Kích thước bất kỳ (sẽ được resize về 512x512 khi training)

OUTPUT:
    - Metadata JSON files trong project/data/food-101N/
    - Cấu trúc dữ liệu sẵn sàng cho training
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2


class Food101NPreprocessor:
    """
    Tiền xử lý dataset Food-101N cho Classification
    
    Paths:
        - Source: KLTN/food-101N/
        - Output: KLTN/project/data/food-101N/
        - Scripts: KLTN/project/classification/
    """
    
    def __init__(self, 
                 source_dir=None,
                 output_dir=None):
        """
        Args:
            source_dir: Thư mục chứa dataset gốc (None = tự động tìm)
            output_dir: Thư mục lưu dữ liệu đã preprocess (None = tự động tìm)
        """
        # Tự động xác định paths dựa trên vị trí script
        script_dir = Path(__file__).parent  # classification/
        project_root = script_dir.parent  # project/
        kltn_root = project_root.parent  # KLTN/
        
        if source_dir is None:
            source_dir = kltn_root / 'food-101N'
        if output_dir is None:
            output_dir = project_root / 'data' / 'food-101N'
            
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # Source paths
        self.images_dir = self.source_dir / 'images'
        self.meta_dir = self.source_dir / 'meta'
        
        # Output paths
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data structures
        self.class_map = {}
        self.dataset_stats = {}
        self.train_data = []
        self.val_data = []
        
        print("=" * 80)
        print("Food-101N Preprocessor Initialized")
        print("=" * 80)
        print(f"📂 Source: {self.source_dir}")
        print(f"📂 Output: {self.output_dir}")
        
    def load_classes(self):
        """Load danh sách 101 classes"""
        print("\n" + "=" * 80)
        print("BƯỚC 1: Load Classes")
        print("=" * 80)
        
        classes_file = self.meta_dir / 'classes.txt'
        
        with open(classes_file, 'r', encoding='utf-8') as f:
            # Skip header
            lines = f.readlines()[1:]
            class_names = [line.strip() for line in lines if line.strip()]
        
        # Tạo class map
        self.class_map = {
            'by_id': {str(i): name for i, name in enumerate(class_names)},
            'by_name': {name: i for i, name in enumerate(class_names)},
            'num_classes': len(class_names)
        }
        
        print(f"\n✅ Loaded {len(class_names)} classes")
        print(f"\n📋 First 10 classes:")
        for i in range(min(10, len(class_names))):
            print(f"  [{i:3d}] {class_names[i]}")
        
        return self.class_map
    
    def load_verified_data(self):
        """
        Load dữ liệu train/val với verification labels
        
        verification_label:
            - 1 = Correct label (ảnh đúng với class)
            - 0 = Incorrect label (ảnh sai - noisy label)
        """
        print("\n" + "=" * 80)
        print("BƯỚC 2: Load Verified Train/Val Data")
        print("=" * 80)
        
        # Load train data
        train_file = self.meta_dir / 'verified_train.tsv'
        print(f"\n📄 Loading {train_file.name}...")
        
        train_correct = 0
        train_noisy = 0
        
        with open(train_file, 'r', encoding='utf-8') as f:
            # Skip header
            lines = f.readlines()[1:]
            
            for line in tqdm(lines, desc="Processing train"):
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                
                img_path, verification = parts
                verification = int(verification)
                
                # Parse class_name/img_key
                class_name, img_key = img_path.split('/')
                class_id = self.class_map['by_name'][class_name]
                
                # Full path
                full_path = str(self.images_dir / class_name / img_key)
                
                self.train_data.append({
                    'image_path': full_path,
                    'class_id': class_id,
                    'class_name': class_name,
                    'verified': verification,
                    'img_key': img_key
                })
                
                if verification == 1:
                    train_correct += 1
                else:
                    train_noisy += 1
        
        print(f"  ✅ Train: {len(self.train_data)} samples")
        print(f"     - Correct: {train_correct} ({train_correct/len(self.train_data)*100:.1f}%)")
        print(f"     - Noisy: {train_noisy} ({train_noisy/len(self.train_data)*100:.1f}%)")
        
        # Load val data
        val_file = self.meta_dir / 'verified_val.tsv'
        print(f"\n📄 Loading {val_file.name}...")
        
        val_correct = 0
        val_noisy = 0
        
        with open(val_file, 'r', encoding='utf-8') as f:
            # Skip header
            lines = f.readlines()[1:]
            
            for line in tqdm(lines, desc="Processing val"):
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                
                img_path, verification = parts
                verification = int(verification)
                
                # Parse class_name/img_key
                class_name, img_key = img_path.split('/')
                class_id = self.class_map['by_name'][class_name]
                
                # Full path
                full_path = str(self.images_dir / class_name / img_key)
                
                self.val_data.append({
                    'image_path': full_path,
                    'class_id': class_id,
                    'class_name': class_name,
                    'verified': verification,
                    'img_key': img_key
                })
                
                if verification == 1:
                    val_correct += 1
                else:
                    val_noisy += 1
        
        print(f"  ✅ Val: {len(self.val_data)} samples")
        print(f"     - Correct: {val_correct} ({val_correct/len(self.val_data)*100:.1f}%)")
        print(f"     - Noisy: {val_noisy} ({val_noisy/len(self.val_data)*100:.1f}%)")
    
    def create_clean_splits(self):
        """
        Tạo clean train/val splits (chỉ giữ verified=1)
        
        Returns:
            train_clean, val_clean
        """
        print("\n" + "=" * 80)
        print("BƯỚC 3: Create Clean Splits (verified=1 only)")
        print("=" * 80)
        
        self.train_clean = [item for item in self.train_data if item['verified'] == 1]
        self.val_clean = [item for item in self.val_data if item['verified'] == 1]
        
        print(f"\n✅ Clean splits created:")
        print(f"  - Train clean: {len(self.train_clean)} / {len(self.train_data)} samples")
        print(f"  - Val clean: {len(self.val_clean)} / {len(self.val_data)} samples")
        
        return self.train_clean, self.val_clean
    
    def analyze_dataset(self):
        """Phân tích thống kê dataset"""
        print("\n" + "=" * 80)
        print("BƯỚC 4: Analyze Dataset Statistics")
        print("=" * 80)
        
        # Count samples per class (train)
        train_class_counts = defaultdict(int)
        train_clean_class_counts = defaultdict(int)
        
        for item in self.train_data:
            train_class_counts[item['class_name']] += 1
            if item['verified'] == 1:
                train_clean_class_counts[item['class_name']] += 1
        
        # Count samples per class (val)
        val_class_counts = defaultdict(int)
        val_clean_class_counts = defaultdict(int)
        
        for item in self.val_data:
            val_class_counts[item['class_name']] += 1
            if item['verified'] == 1:
                val_clean_class_counts[item['class_name']] += 1
        
        # Build stats
        class_stats = []
        for class_name in self.class_map['by_name'].keys():
            class_id = self.class_map['by_name'][class_name]
            
            class_stats.append({
                'class_id': class_id,
                'class_name': class_name,
                'train_total': train_class_counts[class_name],
                'train_clean': train_clean_class_counts[class_name],
                'val_total': val_class_counts[class_name],
                'val_clean': val_clean_class_counts[class_name],
                'total': train_class_counts[class_name] + val_class_counts[class_name]
            })
        
        # Overall stats
        self.dataset_stats = {
            'num_classes': self.class_map['num_classes'],
            'train': {
                'total': len(self.train_data),
                'clean': len(self.train_clean),
                'noisy': len(self.train_data) - len(self.train_clean),
                'clean_ratio': len(self.train_clean) / len(self.train_data)
            },
            'val': {
                'total': len(self.val_data),
                'clean': len(self.val_clean),
                'noisy': len(self.val_data) - len(self.val_clean),
                'clean_ratio': len(self.val_clean) / len(self.val_data)
            },
            'total_images': len(self.train_data) + len(self.val_data),
            'class_stats': class_stats
        }
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total classes: {self.dataset_stats['num_classes']}")
        print(f"  Total images: {self.dataset_stats['total_images']:,}")
        print(f"\n  Training set:")
        print(f"    - Total: {self.dataset_stats['train']['total']:,}")
        print(f"    - Clean: {self.dataset_stats['train']['clean']:,} ({self.dataset_stats['train']['clean_ratio']*100:.1f}%)")
        print(f"    - Noisy: {self.dataset_stats['train']['noisy']:,}")
        print(f"\n  Validation set:")
        print(f"    - Total: {self.dataset_stats['val']['total']:,}")
        print(f"    - Clean: {self.dataset_stats['val']['clean']:,} ({self.dataset_stats['val']['clean_ratio']*100:.1f}%)")
        print(f"    - Noisy: {self.dataset_stats['val']['noisy']:,}")
        
        # Show some class stats
        print(f"\n📋 Sample class statistics:")
        for cs in class_stats[:10]:
            print(f"  [{cs['class_id']:3d}] {cs['class_name']:30s} - "
                  f"Train: {cs['train_clean']}/{cs['train_total']}, "
                  f"Val: {cs['val_clean']}/{cs['val_total']}")
        
        return self.dataset_stats
    
    def verify_images(self):
        """Kiểm tra xem các ảnh có tồn tại và đọc được không"""
        print("\n" + "=" * 80)
        print("BƯỚC 5: Verify Images")
        print("=" * 80)
        
        print("\n🔍 Checking train images...")
        missing_train = []
        corrupted_train = []
        
        for item in tqdm(self.train_data[:100], desc="Sample check train"):  # Check 100 samples
            if not Path(item['image_path']).exists():
                missing_train.append(item['image_path'])
            else:
                try:
                    img = Image.open(item['image_path'])
                    img.verify()
                except:
                    corrupted_train.append(item['image_path'])
        
        print("\n🔍 Checking val images...")
        missing_val = []
        corrupted_val = []
        
        for item in tqdm(self.val_data[:100], desc="Sample check val"):  # Check 100 samples
            if not Path(item['image_path']).exists():
                missing_val.append(item['image_path'])
            else:
                try:
                    img = Image.open(item['image_path'])
                    img.verify()
                except:
                    corrupted_val.append(item['image_path'])
        
        print(f"\n✅ Image verification (sample of 100):")
        print(f"  Train - Missing: {len(missing_train)}, Corrupted: {len(corrupted_train)}")
        print(f"  Val - Missing: {len(missing_val)}, Corrupted: {len(corrupted_val)}")
        
        if missing_train or corrupted_train or missing_val or corrupted_val:
            print(f"\n⚠️  Found some issues (normal for sample check)")
    
    def save_metadata(self):
        """Lưu tất cả metadata"""
        print("\n" + "=" * 80)
        print("BƯỚC 6: Save Metadata")
        print("=" * 80)
        
        # 1. Class map
        class_map_path = self.output_dir / 'class_map.json'
        with open(class_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.class_map, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved: {class_map_path.name}")
        
        # 2. Dataset stats
        stats_path = self.output_dir / 'dataset_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.dataset_stats, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {stats_path.name}")
        
        # 3. Train data (all - with verified labels)
        train_all_path = self.output_dir / 'train_all.json'
        with open(train_all_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {train_all_path.name} ({len(self.train_data)} samples)")
        
        # 4. Train data (clean only)
        train_clean_path = self.output_dir / 'train_clean.json'
        with open(train_clean_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_clean, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {train_clean_path.name} ({len(self.train_clean)} samples)")
        
        # 5. Val data (all - with verified labels)
        val_all_path = self.output_dir / 'val_all.json'
        with open(val_all_path, 'w', encoding='utf-8') as f:
            json.dump(self.val_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {val_all_path.name} ({len(self.val_data)} samples)")
        
        # 6. Val data (clean only)
        val_clean_path = self.output_dir / 'val_clean.json'
        with open(val_clean_path, 'w', encoding='utf-8') as f:
            json.dump(self.val_clean, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {val_clean_path.name} ({len(self.val_clean)} samples)")
        
        print(f"\n📂 All metadata saved to: {self.output_dir}")
    
    def create_image_stats(self, num_samples=1000):
        """Phân tích kích thước ảnh (sample)"""
        print("\n" + "=" * 80)
        print("BƯỚC 7: Analyze Image Dimensions")
        print("=" * 80)
        
        widths = []
        heights = []
        aspect_ratios = []
        
        # Sample random images
        import random
        sample_data = random.sample(self.train_data, min(num_samples, len(self.train_data)))
        
        for item in tqdm(sample_data, desc="Analyzing images"):
            try:
                img = Image.open(item['image_path'])
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w/h)
            except:
                continue
        
        image_stats = {
            'num_analyzed': len(widths),
            'width': {
                'min': int(np.min(widths)),
                'max': int(np.max(widths)),
                'mean': float(np.mean(widths)),
                'median': float(np.median(widths))
            },
            'height': {
                'min': int(np.min(heights)),
                'max': int(np.max(heights)),
                'mean': float(np.mean(heights)),
                'median': float(np.median(heights))
            },
            'aspect_ratio': {
                'min': float(np.min(aspect_ratios)),
                'max': float(np.max(aspect_ratios)),
                'mean': float(np.mean(aspect_ratios)),
                'median': float(np.median(aspect_ratios))
            }
        }
        
        # Save
        image_stats_path = self.output_dir / 'image_stats.json'
        with open(image_stats_path, 'w', encoding='utf-8') as f:
            json.dump(image_stats, f, indent=2)
        
        print(f"\n📊 Image Statistics (from {len(widths)} samples):")
        print(f"  Width:  {image_stats['width']['min']}-{image_stats['width']['max']} "
              f"(avg: {image_stats['width']['mean']:.0f})")
        print(f"  Height: {image_stats['height']['min']}-{image_stats['height']['max']} "
              f"(avg: {image_stats['height']['mean']:.0f})")
        print(f"  Aspect ratio: {image_stats['aspect_ratio']['min']:.2f}-{image_stats['aspect_ratio']['max']:.2f} "
              f"(avg: {image_stats['aspect_ratio']['mean']:.2f})")
        print(f"\n✅ Saved: {image_stats_path.name}")
        
        return image_stats
    
    def run_full_pipeline(self):
        """Chạy toàn bộ pipeline"""
        print("\n" + "=" * 80)
        print("FOOD-101N PREPROCESSING PIPELINE")
        print("=" * 80)
        
        # 1. Load classes
        self.load_classes()
        
        # 2. Load verified data
        self.load_verified_data()
        
        # 3. Create clean splits
        self.create_clean_splits()
        
        # 4. Analyze dataset
        self.analyze_dataset()
        
        # 5. Verify images (sample)
        self.verify_images()
        
        # 6. Save metadata
        self.save_metadata()
        
        # 7. Image stats
        self.create_image_stats()
        
        print("\n" + "=" * 80)
        print("✅ PREPROCESSING HOÀN TẤT!")
        print("=" * 80)
        
        print(f"\n📂 Output directory: {self.output_dir}")
        print("\n📄 Files đã tạo:")
        print("  1. class_map.json       - Mapping class_id <-> class_name (101 classes)")
        print("  2. dataset_stats.json   - Thống kê tổng quan dataset")
        print("  3. image_stats.json     - Thống kê kích thước ảnh")
        print("  4. train_all.json       - Train set (all, có verified labels)")
        print("  5. train_clean.json     - Train set (chỉ verified=1)")
        print("  6. val_all.json         - Val set (all, có verified labels)")
        print("  7. val_clean.json       - Val set (chỉ verified=1)")
        
        print("\n💡 NEXT STEPS:")
        print("  1. Sử dụng train_clean.json / val_clean.json cho training")
        print("  2. Hoặc sử dụng train_all.json và apply noise handling techniques")
        print("  3. Tạo Dataset class và DataLoader")
        print("  4. Train model classification")
        
        print("\n📊 Summary:")
        print(f"  - Classes: {self.class_map['num_classes']}")
        print(f"  - Train (clean): {len(self.train_clean):,} samples")
        print(f"  - Val (clean): {len(self.val_clean):,} samples")
        print(f"  - Total clean: {len(self.train_clean) + len(self.val_clean):,} samples")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Khởi tạo preprocessor (sử dụng relative paths tự động)
    preprocessor = Food101NPreprocessor()
    
    # Chạy full pipeline
    preprocessor.run_full_pipeline()
    
    print("\n" + "=" * 80)
    print("PREPROCESSING PIPELINE COMPLETED!")
    print("=" * 80)
