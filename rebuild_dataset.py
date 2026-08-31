"""
Cashcrow Smartbin — High-Speed Multi-Core Dataset Builder & Label Generator.

Processes the real TrashNet dataset (2,527 images) into YOLO detection format:
- 5 Classes: plastic (0), paper (1), metal (2), glass (3), other (4)
- Stratified 70% Train / 20% Val / 10% Test split
- Ultra-fast GrabCut segmentation (scaled mask computation)
- Generates data/dataset.yaml
"""

import os
import cv2
import json
import random
import shutil
import logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DatasetBuilder")

CLASSES = ["plastic", "paper", "metal", "glass", "other"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

TRASHNET_TO_CASHCROW = {
    "plastic": "plastic",
    "paper": "paper",
    "cardboard": "paper",
    "metal": "metal",
    "glass": "glass",
    "trash": "other",
}


def extract_accurate_bbox(img: np.ndarray) -> tuple:
    """
    Extract a tight bounding box around the object against TrashNet's plain background.
    Uses downscaled GrabCut (256x192) for sub-50ms execution per image with exact normalized coordinates.
    Returns (cx, cy, w, h) in normalized YOLO format [0, 1].
    """
    orig_h, orig_w = img.shape[:2]
    
    # Downscale for ultra-fast GrabCut computation
    target_w, target_h = 256, 192
    small_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    margin_x = max(6, int(target_w * 0.04))
    margin_y = max(6, int(target_h * 0.04))
    rect = (margin_x, margin_y, target_w - 2 * margin_x, target_h - 2 * margin_y)

    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(small_img, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest)
            area_frac = (bw * bh) / (target_w * target_h)
            if 0.03 <= area_frac <= 0.95 and bw > 10 and bh > 10:
                pad_x = int(bw * 0.02)
                pad_y = int(bh * 0.02)
                bx = max(0, bx - pad_x)
                by = max(0, by - pad_y)
                bw = min(target_w - bx, bw + 2 * pad_x)
                bh = min(target_h - by, bh + 2 * pad_y)
                return ((bx + bw / 2.0) / target_w, (by + bh / 2.0) / target_h, bw / target_w, bh / target_h, "grabcut")
    except Exception:
        pass

    # Method 2: Color background difference
    try:
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        corner_pixels = np.concatenate([
            gray[:15, :15].flatten(),
            gray[:15, -15:].flatten(),
            gray[-15:, :15].flatten(),
            gray[-15:, -15:].flatten(),
        ])
        bg_val = np.median(corner_pixels)
        diff = cv2.absdiff(gray, int(bg_val))
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest)
            area_frac = (bw * bh) / (target_w * target_h)
            if 0.03 <= area_frac <= 0.95:
                return ((bx + bw / 2.0) / target_w, (by + bh / 2.0) / target_h, bw / target_w, bh / target_h, "thresh")
    except Exception:
        pass

    # Fallback: 75% center crop
    return (0.5, 0.5, 0.75, 0.75, "fallback")


def _process_single_image(args):
    split, cls_name, cls_idx, src_path_str, output_dir_str = args
    src_path = Path(src_path_str)
    output_dir = Path(output_dir_str)
    
    img = cv2.imread(str(src_path))
    if img is None or img.size == 0:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    cx, cy, nw, nh, method = extract_accurate_bbox(img)

    img_filename = f"{cls_name}_{src_path.stem}.jpg"
    dest_img = output_dir / "images" / split / img_filename
    dest_lbl = output_dir / "labels" / split / f"{cls_name}_{src_path.stem}.txt"

    cv2.imwrite(str(dest_img), img)
    with open(dest_lbl, "w") as f:
        f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    return (split, cls_name, method, blur_score)


def rebuild_real_dataset(
    trashnet_src: Path = Path("data/trashnet_extracted/dataset-resized"),
    output_dir: Path = Path("data/dataset"),
    seed: int = 42,
    num_workers: int = 10,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    if not trashnet_src.exists():
        raise FileNotFoundError(f"TrashNet source folder not found at {trashnet_src}")

    if output_dir.exists():
        logger.info("Cleaning existing dataset at %s...", output_dir)
        shutil.rmtree(output_dir)

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    grouped_images = defaultdict(list)
    total_found = 0

    for folder_name in sorted(os.listdir(trashnet_src)):
        folder_path = trashnet_src / folder_name
        if not folder_path.is_dir() or folder_name.startswith("."):
            continue

        target_class = TRASHNET_TO_CASHCROW.get(folder_name)
        if not target_class:
            continue

        files = [
            folder_path / f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        grouped_images[target_class].extend(files)
        total_found += len(files)

    logger.info("Total images indexed: %d across 5 target classes", total_found)

    tasks = []
    for cls_name, img_paths in grouped_images.items():
        cls_idx = CLASS_TO_IDX[cls_name]
        random.shuffle(img_paths)
        n = len(img_paths)
        n_train = int(n * 0.70)
        n_val = int(n * 0.20)

        for p in img_paths[:n_train]:
            tasks.append(("train", cls_name, cls_idx, str(p), str(output_dir)))
        for p in img_paths[n_train:n_train + n_val]:
            tasks.append(("val", cls_name, cls_idx, str(p), str(output_dir)))
        for p in img_paths[n_train + n_val:]:
            tasks.append(("test", cls_name, cls_idx, str(p), str(output_dir)))

    logger.info("Starting multi-core processing with %d workers on %d images...", num_workers, len(tasks))

    stats = {
        "splits": {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)},
        "bbox_methods": defaultdict(int),
        "blur_scores": [],
        "processed": 0,
        "failed": 0,
    }

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_single_image, t) for t in tasks]
        done_count = 0
        for fut in as_completed(futures):
            res = fut.result()
            done_count += 1
            if res is not None:
                split, cls_name, method, blur_val = res
                stats["splits"][split][cls_name] += 1
                stats["bbox_methods"][method] += 1
                stats["blur_scores"].append(blur_val)
                stats["processed"] += 1
            else:
                stats["failed"] += 1
            if done_count % 500 == 0 or done_count == len(tasks):
                logger.info("  Processed %d / %d images...", done_count, len(tasks))

    # Write data/dataset.yaml
    yaml_path = Path("data/dataset.yaml")
    yaml_content = f"""path: {output_dir.absolute().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: plastic
  1: paper
  2: metal
  3: glass
  4: other
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    logger.info("Updated %s with 5 classes and train/val/test splits.", yaml_path)

    # Print Summary Table
    print("\n" + "=" * 70)
    print("                    DATASET GENERATION SUMMARY REPORT                  ")
    print("=" * 70)
    print(f"{'Class ID':<10} {'Class Name':<12} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 70)
    for idx, name in enumerate(CLASSES):
        tr = stats["splits"]["train"][name]
        va = stats["splits"]["val"][name]
        te = stats["splits"]["test"][name]
        tot = tr + va + te
        print(f"{idx:<10} {name:<12} {tr:<10} {va:<10} {te:<10} {tot:<10}")
    print("-" * 70)
    total_tr = sum(stats["splits"]["train"].values())
    total_va = sum(stats["splits"]["val"].values())
    total_te = sum(stats["splits"]["test"].values())
    print(f"{'TOTAL':<22} {total_tr:<10} {total_va:<10} {total_te:<10} {stats['processed']:<10}")
    print("=" * 70)
    print(f"Bounding box methods used: {dict(stats['bbox_methods'])}")
    if stats["blur_scores"]:
        print(f"Mean image blur score: {np.mean(stats['blur_scores']):.2f} (std={np.std(stats['blur_scores']):.2f})")
    print("=" * 70 + "\n")

    return stats


if __name__ == "__main__":
    rebuild_real_dataset()
