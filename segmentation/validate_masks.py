import csv
from pathlib import Path
import numpy as np
import cv2

base = Path(r"D:\Dev\University\KLTN\project\data\foodseg103")

for split in ('train','val','test'):
    manifest = base / f'manifest_{split}.csv'
    total = 0
    nonempty = 0
    if not manifest.exists():
        print(split, "no manifest")
        continue
    with manifest.open('r', encoding='utf-8') as f:
        next(f)  # header
        for line in f:
            total += 1
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            mask_path = parts[1]
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            if np.any(mask != 0):
                nonempty += 1
    print(f"{split}: total={total}, non-empty masks={nonempty}, fraction={nonempty/total:.3f}")
