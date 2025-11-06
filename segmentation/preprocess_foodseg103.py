import os
import cv2
import json
import numpy as np
import base64
import zlib
import gzip
from io import BytesIO
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import csv
from collections import defaultdict


# Default parameters
DEFAULT_SRC = r'D:\Dev\University\KLTN\foodseg103'
DEFAULT_DST = r'D:\Dev\University\KLTN\project\data\foodseg103'
DEFAULT_IMG_SIZE = (512, 512)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess foodseg103 dataset into image+mask pairs')
    parser.add_argument('--src', type=str, default=DEFAULT_SRC, help='Path to original foodseg103 folder')
    parser.add_argument('--dst', type=str, default=DEFAULT_DST, help='Output folder for preprocessed data')
    parser.add_argument('--size', type=int, nargs=2, default=DEFAULT_IMG_SIZE, help='Output image size W H')
    parser.add_argument('--val-size', type=float, default=0.15, help='Validation split fraction from train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode: save failed decoded bitmaps and extra logs')
    parser.add_argument('--debug-dir', type=str, default=None, help='Directory to save debug artifacts (defaults to dst/debug)')
    return parser.parse_args()


def load_meta(src_root):
    meta_path = os.path.join(src_root, 'meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'meta.json not found at {meta_path}')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    # Build mappings: index from 1..N, 0 reserved for background
    classes = meta.get('classes', [])
    class_map = {}
    id_to_index = {}
    name_to_index = {}
    for i, cls in enumerate(classes):
        idx = i + 1
        # Prefer 'title' (human readable) then 'name', fallback to id
        name = cls.get('title') or cls.get('name') or str(cls.get('id'))
        cid = cls.get('id')
        # store friendly name -> idx
        class_map[name] = idx
        # store id mapping for both int and str keys to be robust
        if cid is not None:
            id_to_index[cid] = idx
            id_to_index[str(cid)] = idx
        name_to_index[name] = idx

    return class_map, id_to_index, name_to_index


def save_class_map(dst_root, class_map, id_to_index):
    os.makedirs(dst_root, exist_ok=True)
    out = {
        'by_name': class_map,
        'by_id': {str(k): v for k, v in id_to_index.items()}
    }
    with open(os.path.join(dst_root, 'class_map.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def bitmap_to_mask(bitmap_data, mask_h, mask_w, offset_x, offset_y, label_id):
    """Decode base64 bitmap (image) and place into a full-size mask array at offset.
    Returns a mask array of shape (mask_h, mask_w) with 0/label_id.
    """
    try:
        data_bytes = base64.b64decode(bitmap_data)
        img_buffer = np.frombuffer(data_bytes, np.uint8)
        bitmap_img = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
        if bitmap_img is None:
            # try zlib decompress (Label-Studio stores zlib-compressed PNG bytes sometimes)
            try:
                dec = zlib.decompress(data_bytes)
                img_buffer = np.frombuffer(dec, np.uint8)
                bitmap_img = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
                if bitmap_img is not None:
                    # success
                    pass
            except Exception:
                bitmap_img = None
        if bitmap_img is None:
            # try gzip
            try:
                dec = gzip.decompress(data_bytes)
                img_buffer = np.frombuffer(dec, np.uint8)
                bitmap_img = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
            except Exception:
                bitmap_img = None
    except Exception:
        return None
    if bitmap_img is None:
        return None

    # Convert to single-channel mask
    if len(bitmap_img.shape) > 2:
        # If RGBA, alpha channel often indicates mask
        if bitmap_img.shape[2] == 4:
            bitmap_mask = bitmap_img[:, :, 3]
        else:
            # try first channel
            bitmap_mask = bitmap_img[:, :, 0]
    else:
        bitmap_mask = bitmap_img

    h, w = bitmap_mask.shape
    full_mask = np.zeros((mask_h, mask_w), dtype=np.uint16 if label_id > 255 else np.uint8)

    # Compute placement coordinates and clamp
    y1 = max(0, int(offset_y))
    x1 = max(0, int(offset_x))
    y2 = min(mask_h, y1 + h)
    x2 = min(mask_w, x1 + w)

    src_y1 = 0
    src_x1 = 0
    if y2 - y1 <= 0 or x2 - x1 <= 0:
        return full_mask

    # If bitmap was partially outside, adjust source slice
    if y1 + h > mask_h:
        src_y2 = mask_h - y1
    else:
        src_y2 = h
    if x1 + w > mask_w:
        src_x2 = mask_w - x1
    else:
        src_x2 = w

    src_slice = bitmap_mask[src_y1:src_y2, src_x1:src_x2]
    mask_pixels = (src_slice > 0)
    full_mask[y1:y1+mask_pixels.shape[0], x1:x1+mask_pixels.shape[1]][mask_pixels] = label_id

    return full_mask


def polygon_to_mask(points, mask_h, mask_w, label_id):
    pts = np.array(points, dtype=np.int32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    mask = np.zeros((mask_h, mask_w), dtype=np.uint16 if label_id > 255 else np.uint8)
    try:
        cv2.fillPoly(mask, [pts], color=int(label_id))
    except Exception:
        pass
    return mask


def ensure_dirs(dst_root, splits):
    for s in splits:
        os.makedirs(os.path.join(dst_root, 'images', s), exist_ok=True)
        os.makedirs(os.path.join(dst_root, 'annotations', s), exist_ok=True)


def process_split(split_name, img_list, src_subdir, dst_root, img_size, id_to_index, name_to_index, debug=False, debug_dir=None):
    src_img_dir = os.path.join(src_subdir, 'img')
    src_ann_dir = os.path.join(src_subdir, 'ann')

    images_out = os.path.join(dst_root, 'images', split_name)
    ann_out = os.path.join(dst_root, 'annotations', split_name)
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(ann_out, exist_ok=True)

    manifest_rows = []
    skipped = 0
    processed = 0
    labels_seen = set()

    for fname in tqdm(img_list, desc=f'Processing {split_name}'):
        name_noext = os.path.splitext(fname)[0]
        img_path = os.path.join(src_img_dir, fname)
        ann_path = os.path.join(src_ann_dir, fname + '.json')

        if not os.path.exists(img_path) or not os.path.exists(ann_path):
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]
        # Choose mask dtype depending on number of classes
        composite_mask = np.zeros((img_h, img_w), dtype=np.uint16)

        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
        except Exception:
            skipped += 1
            continue

        for obj in ann.get('objects', []):
            geom = obj.get('geometryType')
            label_id = 0
            # classId may be numeric or string; first try id mapping then name
            cid = obj.get('classId')
            if cid in id_to_index:
                label_id = id_to_index[cid]
            else:
                # try by name
                title = obj.get('classTitle') or obj.get('label') or obj.get('className')
                if title in name_to_index:
                    label_id = name_to_index[title]

            if label_id == 0:
                # unknown class -> skip object
                continue

            # handle bitmap
            if geom == 'bitmap' and 'bitmap' in obj:
                bd = obj['bitmap']
                bitmap_data = bd.get('data')
                origin = bd.get('origin', [0, 0])
                ox, oy = origin[0], origin[1]
                obj_mask = None
                if bitmap_data is not None:
                    obj_mask = bitmap_to_mask(bitmap_data, img_h, img_w, ox, oy, label_id)
                if obj_mask is None:
                    # try alternate structure
                    # if bd itself is a buffer-like structure, this may fail; so handle at caller
                    try:
                        obj_mask = bitmap_to_mask(bd, img_h, img_w, 0, 0, label_id)
                    except Exception:
                        obj_mask = None
                if obj_mask is not None:
                    # overwrite existing pixels with new label
                    mask_pixels = (obj_mask > 0)
                    composite_mask[mask_pixels] = obj_mask[mask_pixels]
                else:
                    # debug: save the raw base64 and try to decode/save the bitmap for inspection
                    if debug and bitmap_data is not None:
                        os.makedirs(debug_dir, exist_ok=True)
                        try:
                            data_bytes = base64.b64decode(bitmap_data)
                            buf = np.frombuffer(data_bytes, np.uint8)
                            dec = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                            if dec is not None:
                                # save decoded bitmap (may have alpha)
                                dbg_path = os.path.join(debug_dir, f"{split_name}_{name_noext}_obj{obj.get('id', 'x')}.png")
                                cv2.imwrite(dbg_path, dec)
                            else:
                                # save base64 text for later inspection
                                dbg_txt = os.path.join(debug_dir, f"{split_name}_{name_noext}_obj{obj.get('id', 'x')}.b64.txt")
                                with open(dbg_txt, 'w', encoding='utf-8') as bf:
                                    bf.write(bitmap_data)
                        except Exception:
                            try:
                                dbg_txt = os.path.join(debug_dir, f"{split_name}_{name_noext}_obj{obj.get('id', 'x')}.b64.txt")
                                with open(dbg_txt, 'w', encoding='utf-8') as bf:
                                    bf.write(str(bitmap_data))
                            except Exception:
                                pass

            # handle polygon-like
            elif geom in ('polygon', 'polyline') or 'points' in obj:
                pts = obj.get('points') or obj.get('polygon') or obj.get('path')
                if pts:
                    # pts may be list of [x,y] or {'x':..,'y':..}
                    if isinstance(pts, list) and len(pts) and isinstance(pts[0], dict):
                        pts = [[int(p['x']), int(p['y'])] for p in pts]
                    elif isinstance(pts, list) and len(pts) and isinstance(pts[0], list):
                        pts = [[int(p[0]), int(p[1])] for p in pts]
                    try:
                        obj_mask = polygon_to_mask(pts, img_h, img_w, label_id)
                        mask_pixels = (obj_mask > 0)
                        composite_mask[mask_pixels] = obj_mask[mask_pixels]
                    except Exception:
                        pass

        # Resize and save
        img_resized = cv2.resize(img, tuple(img_size), interpolation=cv2.INTER_AREA)
        # If composite_mask dtype is uint16 and img_size small, keep dtype
        mask_resized = cv2.resize(composite_mask, tuple(img_size), interpolation=cv2.INTER_NEAREST)

        out_img_path = os.path.join(images_out, fname)
        out_mask_path = os.path.join(ann_out, f'{name_noext}.png')
        cv2.imwrite(out_img_path, img_resized)
        # Ensure mask dtype fits PNG
        if mask_resized.max() <= 255:
            cv2.imwrite(out_mask_path, mask_resized.astype(np.uint8))
        else:
            cv2.imwrite(out_mask_path, mask_resized.astype(np.uint16))

        unique = np.unique(mask_resized)
        labels_seen.update([int(u) for u in unique if u != 0])
        manifest_rows.append((out_img_path, out_mask_path, img_resized.shape[1], img_resized.shape[0], ';'.join(map(str, unique.tolist()))))
        processed += 1

    # Write manifest
    manifest_path = os.path.join(dst_root, f'manifest_{split_name}.csv')
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'mask_path', 'width', 'height', 'unique_labels'])
        for r in manifest_rows:
            writer.writerow(r)

    print(f'Finished {split_name}: processed={processed}, skipped={skipped}, labels_seen={sorted(labels_seen)}')
    return processed, skipped, labels_seen


def main():
    args = parse_args()
    SRC_ROOT = args.src
    DST_ROOT = args.dst
    IMG_SIZE = tuple(args.size)
    DEBUG = args.debug
    DEBUG_DIR = args.debug_dir or os.path.join(DST_ROOT, 'debug_bitmaps')

    # load meta and mappings
    class_map, id_to_index, name_to_index = load_meta(SRC_ROOT)
    save_class_map(DST_ROOT, class_map, id_to_index)

    # prepare split lists
    splits_src = {
        'train': os.path.join(SRC_ROOT, 'train'),
        'test': os.path.join(SRC_ROOT, 'test')
    }

    train_img_dir = os.path.join(splits_src['train'], 'img')
    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f'Train image dir not found: {train_img_dir}')
    valid_ext = ('.jpg', '.jpeg', '.png')
    all_train_files = sorted([f for f in os.listdir(train_img_dir) if f.lower().endswith(valid_ext)])
    train_files, val_files = train_test_split(all_train_files, test_size=args.val_size, random_state=args.seed)

    test_img_dir = os.path.join(splits_src['test'], 'img')
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f'Test image dir not found: {test_img_dir}')
    all_test_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(valid_ext)])

    splits = {'train': train_files, 'val': val_files, 'test': all_test_files}
    print(f"Split sizes: train={len(train_files)}, val={len(val_files)}, test={len(all_test_files)}")

    ensure_dirs(DST_ROOT, splits.keys())

    total_labels = set()
    # process each split
    for s in ['train', 'val', 'test']:
        src_subdir = splits_src['train'] if s in ('train', 'val') else splits_src['test']
        p, sk, labels = process_split(s, splits[s], src_subdir, DST_ROOT, IMG_SIZE, id_to_index, name_to_index, debug=DEBUG, debug_dir=DEBUG_DIR)
        total_labels.update(labels)

    print(f'All done. labels found across dataset: {sorted(total_labels)}')


if __name__ == '__main__':
    main()
