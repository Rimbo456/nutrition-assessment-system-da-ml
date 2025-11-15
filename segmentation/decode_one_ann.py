"""
Decode bitmap(s) inside one annotation JSON and save decoded bitmaps (or base64 text) for inspection.
Usage:
  python project\decode_one_ann.py --ann "path/to/00000380.jpg.json" --out output_dir

This script does NOT modify dataset; it only writes debug outputs.
"""
import os
import argparse
import json
import base64
import zlib
import gzip
from pathlib import Path
import cv2
import numpy as np


def parse_args():
    # Get default paths relative to project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_ann = os.path.join(os.path.dirname(project_root), "foodseg103", "train", "ann", "00000380.jpg.json")
    default_out = os.path.join(project_root, "temp", "fs_debug_single")
    
    p = argparse.ArgumentParser(description='Decode bitmap entries from one annotation JSON')
    p.add_argument('--ann', type=str, required=False,
                   default=default_ann,
                   help='Path to annotation JSON file')
    p.add_argument('--out', type=str, required=False, default=default_out,
                   help='Directory to save decoded bitmaps / b64 text')
    return p.parse_args()


def main():
    args = parse_args()
    ann_path = Path(args.ann)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ann_path.exists():
        print('Annotation file not found:', ann_path)
        return

    with ann_path.open('r', encoding='utf-8') as f:
        ann = json.load(f)

    print('Annotation size:', ann.get('size'))
    for i, obj in enumerate(ann.get('objects', [])):
        print('\nObject', i, 'id:', obj.get('id'), 'classId:', obj.get('classId'), 'classTitle:', obj.get('classTitle'), 'geometryType:', obj.get('geometryType'))
        if obj.get('geometryType') == 'bitmap' and 'bitmap' in obj:
            bd = obj['bitmap']
            data = bd.get('data')
            origin = bd.get('origin', [0, 0])
            print(' origin:', origin, 'data length:', len(data) if data else None)
            if not data:
                print(' No bitmap data')
                continue
            try:
                b = base64.b64decode(data)

                # Try direct imdecode first
                arr = np.frombuffer(b, np.uint8)
                dec = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if dec is None:
                    # Try zlib decompress (labelstudio often stores zlib-compressed PNG bytes)
                    try:
                        decompressed = zlib.decompress(b)
                        arr2 = np.frombuffer(decompressed, np.uint8)
                        dec = cv2.imdecode(arr2, cv2.IMREAD_UNCHANGED)
                        if dec is not None:
                            print(' -> zlib-decompressed then imdecode succeeded')
                    except Exception:
                        dec = None

                if dec is None:
                    # Try gzip
                    try:
                        decompressed = gzip.decompress(b)
                        arr3 = np.frombuffer(decompressed, np.uint8)
                        dec = cv2.imdecode(arr3, cv2.IMREAD_UNCHANGED)
                        if dec is not None:
                            print(' -> gzip-decompressed then imdecode succeeded')
                    except Exception:
                        dec = None

                if dec is None:
                    print(' -> imdecode returned None after trying zlib/gzip; saving base64 text for inspection')
                    (out_dir / f"{ann_path.stem}_obj{i}.b64.txt").write_text(data, encoding='utf-8')
                else:
                    print(' -> decoded shape:', dec.shape, 'dtype:', dec.dtype)
                    save_path = out_dir / f"{ann_path.stem}_obj{i}.png"
                    # Save as-is to preserve alpha channel if present
                    cv2.imwrite(str(save_path), dec)
                    print(' Saved decoded bitmap to', save_path)
            except Exception as e:
                print(' decode exception:', e)
                try:
                    (out_dir / f"{ann_path.stem}_obj{i}.b64.txt").write_text(str(data)[:2000], encoding='utf-8')
                except Exception:
                    pass
        else:
            print(' Not a bitmap object or missing bitmap field')

    print('\nSaved debug outputs to', out_dir)


if __name__ == '__main__':
    main()
