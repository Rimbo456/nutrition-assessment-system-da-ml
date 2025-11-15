"""
Script để convert absolute paths sang relative paths trong JSON files
Chạy script này nếu bạn đã có JSON files với absolute paths
"""

import json
import os
from pathlib import Path


def convert_to_relative_paths(json_path, base_dir):
    """
    Convert absolute paths trong JSON file thành relative paths
    
    Từ: D:/path/to/KLTN/food-101N/images/class/image.jpg
    Thành: ../../../food-101N/images/class/image.jpg
    
    Args:
        json_path: Đường dẫn đến JSON file (trong data/food-101N/)
        base_dir: Base directory (data/food-101N/)
    """
    print(f"\n📄 Converting {json_path.name}...")
    
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert paths
    converted = 0
    skipped = 0
    
    for item in data:
        old_path = item['image_path']
        
        # Nếu đã là relative path thì skip
        if not Path(old_path).is_absolute():
            skipped += 1
            continue
        
        # Convert to relative path
        try:
            path_obj = Path(old_path)
            parts = path_obj.parts
            
            # Tìm 'food-101N' trong path
            if 'food-101N' in parts:
                food101n_idx = parts.index('food-101N')
                relative_parts = parts[food101n_idx:]
                
                # Tạo relative path: ../../../food-101N/images/...
                rel_path = os.path.join('..', '..', '..', *relative_parts)
                
                # Normalize slashes
                rel_path = rel_path.replace('\\', '/')
                
                item['image_path'] = rel_path
                converted += 1
            else:
                print(f"  ⚠️  Cannot find 'food-101N' in {old_path}")
                
        except Exception as e:
            print(f"  ⚠️  Cannot convert {old_path}: {e}")
    
    # Save back
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Converted: {converted}, Skipped: {skipped}, Total: {len(data)}")
    
    # Show sample
    if len(data) > 0:
        print(f"  📝 Sample path: {data[0]['image_path']}")


def main():
    print("=" * 80)
    print("CONVERT ABSOLUTE PATHS → RELATIVE PATHS")
    print("=" * 80)
    
    # Paths
    script_dir = Path(__file__).parent  # classification/
    data_dir = script_dir.parent / 'data' / 'food-101N'
    
    print(f"\n📂 Data directory: {data_dir}")
    
    # Convert các JSON files
    json_files = [
        'train_all.json',
        'train_clean.json',
        'val_all.json',
        'val_clean.json'
    ]
    
    for json_file in json_files:
        json_path = data_dir / json_file
        
        if not json_path.exists():
            print(f"\n⚠️  {json_file} not found, skipping...")
            continue
        
        convert_to_relative_paths(json_path, data_dir)
    
    print("\n" + "=" * 80)
    print("✅ CONVERSION COMPLETED!")
    print("=" * 80)
    print("\n💡 Bây giờ bạn có thể:")
    print("  1. Copy toàn bộ thư mục KLTN sang máy khác")
    print("  2. Chạy training ngay mà không cần preprocessing lại")
    print("  3. Paths sẽ tự động resolve đúng trên máy mới")


if __name__ == "__main__":
    main()
