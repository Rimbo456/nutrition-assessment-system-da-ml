"""
Create a visual overlay between an original image and a decoded bitmap (mask).
Usage:
  python project\inspect_mask_one.py --img "path/to/img.jpg" --bitmap "path/to/bitmap.png" --out "output.png"

This helps verify origin placement and mask shape.
"""
import os
import argparse
from pathlib import Path
import cv2
import numpy as np


def parse_args():
    # Default output path relative to project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_out = os.path.join(project_root, "temp", "fs_debug_single", "overlay.png")
    
    p = argparse.ArgumentParser(description='Inspect one decoded bitmap overlayed on original image')
    p.add_argument('--img', type=str, required=True, help='Path to original image')
    p.add_argument('--bitmap', type=str, required=True, help='Path to decoded bitmap PNG (may have alpha)')
    p.add_argument('--origin', type=int, nargs=2, required=False, help='Origin (x y) where bitmap should be placed on the image')
    p.add_argument('--out', type=str, required=False, default=default_out, help='Output overlay path')
    return p.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.img)
    bmp_path = Path(args.bitmap)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path))
    if img is None:
        print('Cannot read image:', img_path)
        return
    bmp = cv2.imread(str(bmp_path), cv2.IMREAD_UNCHANGED)
    if bmp is None:
        print('Cannot read bitmap:', bmp_path)
        return

    print('img shape', img.shape, 'bitmap shape', bmp.shape)

    # extract alpha or luminance channel
    alpha = None
    if bmp.ndim == 3 and bmp.shape[2] == 4:
        alpha = bmp[:, :, 3]
    elif bmp.ndim == 2:
        alpha = bmp
    elif bmp.ndim == 3:
        alpha = bmp[:, :, 0]
    else:
        print('Unsupported bitmap format')
        return

    mask = (alpha > 0).astype('uint8') * 255

    # If bitmap size != image size, place bitmap mask into a full-size canvas using origin
    if bmp.shape[0] != img.shape[0] or bmp.shape[1] != img.shape[1]:
        if args.origin is None:
            print('Bitmap size differs from image size and --origin not provided. Cannot place bitmap.')
            return
        ox, oy = int(args.origin[0]), int(args.origin[1])
        img_h, img_w = img.shape[:2]
        canvas_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        # compute placement and clipping
        y1 = oy
        x1 = ox
        h, w = mask.shape
        y2 = min(img_h, y1 + h)
        x2 = min(img_w, x1 + w)
        src_y2 = y2 - y1
        src_x2 = x2 - x1
        if src_y2 <= 0 or src_x2 <= 0:
            print('Origin places bitmap completely outside the image. Nothing to overlay.')
            return
        canvas_mask[y1:y2, x1:x2] = mask[0:src_y2, 0:src_x2]
        mask = canvas_mask

    col = np.zeros_like(img)
    col[mask == 255] = (0, 0, 255)  # red overlay for mask
    overlay = cv2.addWeighted(img, 0.7, col, 0.3, 0)
    cv2.imwrite(str(out_path), overlay)
    print('Saved overlay to', out_path)

if __name__ == '__main__':
    main()
