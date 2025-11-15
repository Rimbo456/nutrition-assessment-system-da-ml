"""
Quick script to regenerate manifest files with relative paths.
Use this if you've already preprocessed data but manifests have absolute paths.
"""
import os
import csv
from pathlib import Path
from tqdm import tqdm


def regenerate_manifests(data_root):
    """
    Scan preprocessed data directory and create manifest files with relative paths.
    
    Args:
        data_root: Path to preprocessed data (e.g., project/data/foodseg103)
    """
    data_root = Path(data_root)
    
    if not data_root.exists():
        raise ValueError(f"Data root does not exist: {data_root}")
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        img_dir = data_root / 'images' / split
        mask_dir = data_root / 'annotations' / split
        manifest_path = data_root / f'manifest_{split}.csv'
        
        if not img_dir.exists() or not mask_dir.exists():
            print(f"⚠️  Skipping {split}: directories not found")
            continue
        
        print(f"\n📋 Generating manifest for {split}...")
        
        # Get all images
        image_files = sorted(img_dir.glob('*.jpg'))
        
        if len(image_files) == 0:
            print(f"⚠️  No images found in {img_dir}")
            continue
        
        manifest_rows = []
        failed = []
        
        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            # Corresponding mask
            mask_path = mask_dir / f"{img_path.stem}.png"
            
            if not mask_path.exists():
                failed.append(img_path.name)
                continue
            
            # Create relative paths from data_root
            rel_img = os.path.relpath(img_path, data_root)
            rel_mask = os.path.relpath(mask_path, data_root)
            
            manifest_rows.append({
                'image_path': rel_img.replace('\\', '/'),  # Use forward slashes for portability
                'mask_path': rel_mask.replace('\\', '/')
            })
        
        # Write manifest
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'mask_path'])
            writer.writeheader()
            writer.writerows(manifest_rows)
        
        print(f"✅ {split}: {len(manifest_rows)} samples written to {manifest_path.name}")
        
        if failed:
            print(f"⚠️  {len(failed)} samples missing masks:")
            for name in failed[:5]:
                print(f"   - {name}")
            if len(failed) > 5:
                print(f"   ... and {len(failed) - 5} more")


def main():
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_root = project_root / 'data' / 'foodseg103'
    
    print("=" * 60)
    print("🔄 Regenerating Manifest Files with Relative Paths")
    print("=" * 60)
    print(f"Data root: {data_root}")
    
    regenerate_manifests(data_root)
    
    print("\n" + "=" * 60)
    print("✅ Done! Verify manifest files:")
    print(f"   head {data_root / 'manifest_train.csv'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
