"""
Cashcrow Smartbin — Dataset Quality & Bounding Box Audit.

Performs deep validation on the generated YOLO dataset:
- Format verification (class IDs, coordinates within [0, 1])
- Per-class area fraction & aspect ratio distribution statistics
- Identification of outlier / suspiciously small or large bounding boxes
- Renders sample annotated images with bounding boxes for visual audit
"""

import os
import cv2
from pathlib import Path
from collections import defaultdict
import numpy as np

CLASSES = ["plastic", "paper", "metal", "glass", "other"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


def audit_dataset_quality(dataset_dir: Path = Path("data/dataset"), sample_output_dir: Path = Path("audit_samples")):
    sample_output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 75)
    print("                    DATASET QUALITY & INTEGRITY AUDIT                   ")
    print("=" * 75)

    stats = {
        "splits": {},
        "corrupt_labels": [],
        "out_of_bounds": [],
        "tiny_boxes": [],
        "huge_boxes": [],
        "class_areas": defaultdict(list),
        "class_aspect_ratios": defaultdict(list),
    }

    samples_saved = defaultdict(int)

    for split in ["train", "val", "test"]:
        lbl_dir = dataset_dir / "labels" / split
        img_dir = dataset_dir / "images" / split

        if not lbl_dir.exists():
            print(f"Warning: split directory {lbl_dir} does not exist.")
            continue

        lbl_files = sorted(lbl_dir.glob("*.txt"))
        img_files = sorted(img_dir.glob("*.jpg"))

        split_counts = defaultdict(int)
        total_boxes = 0

        for lbl_path in lbl_files:
            img_path = img_dir / f"{lbl_path.stem}.jpg"
            if not img_path.exists():
                stats["corrupt_labels"].append(f"Missing image for {lbl_path}")
                continue

            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]

            if not lines:
                stats["corrupt_labels"].append(f"Empty label file: {lbl_path}")
                continue

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    stats["corrupt_labels"].append(f"Malformed line in {lbl_path}: {line}")
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                except ValueError:
                    stats["corrupt_labels"].append(f"Non-numeric values in {lbl_path}: {line}")
                    continue

                if not (0 <= cls_id < len(CLASSES)):
                    stats["corrupt_labels"].append(f"Invalid class ID {cls_id} in {lbl_path}")
                    continue

                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    stats["out_of_bounds"].append(f"Out of bounds in {lbl_path}: {cx}, {cy}, {w}, {h}")
                    continue

                cls_name = CLASSES[cls_id]
                area_frac = w * h
                aspect_ratio = w / h if h > 0 else 0.0

                stats["class_areas"][cls_name].append(area_frac)
                stats["class_aspect_ratios"][cls_name].append(aspect_ratio)

                if area_frac < 0.02:
                    stats["tiny_boxes"].append((lbl_path.name, cls_name, area_frac))
                if area_frac > 0.98:
                    stats["huge_boxes"].append((lbl_path.name, cls_name, area_frac))

                split_counts[cls_name] += 1
                total_boxes += 1

                # Save sample annotated images (2 per class from val split)
                if split == "val" and samples_saved[cls_name] < 2:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        ih, iw = img.shape[:2]
                        x1 = int((cx - w / 2) * iw)
                        y1 = int((cy - h / 2) * ih)
                        x2 = int((cx + w / 2) * iw)
                        y2 = int((cy + h / 2) * ih)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            img,
                            f"{cls_name} ({w*100:.0f}%x{h*100:.0f}%)",
                            (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
                        sample_path = sample_output_dir / f"sample_{cls_name}_{samples_saved[cls_name]}.jpg"
                        cv2.imwrite(str(sample_path), img)
                        samples_saved[cls_name] += 1

        stats["splits"][split] = {
            "image_count": len(img_files),
            "label_count": len(lbl_files),
            "box_count": total_boxes,
            "per_class": dict(split_counts),
        }

    # Print summary
    print(f"Total Splits Audited : {len(stats['splits'])}")
    print(f"Corrupt Labels Found : {len(stats['corrupt_labels'])}")
    print(f"Out of Bounds Boxes  : {len(stats['out_of_bounds'])}")
    print(f"Tiny Boxes (<2% area): {len(stats['tiny_boxes'])}")
    print(f"Huge Boxes (>98% area): {len(stats['huge_boxes'])}")

    print("\n" + "-" * 75)
    print(f"{'Class Name':<12} | {'Count':<7} | {'Median Area %':<15} | {'Min Area %':<12} | {'Max Area %':<12}")
    print("-" * 75)
    for name in CLASSES:
        areas = stats["class_areas"][name]
        if areas:
            med_a = np.median(areas) * 100
            min_a = np.min(areas) * 100
            max_a = np.max(areas) * 100
            print(f"{name:<12} | {len(areas):<7} | {med_a:>13.1f}% | {min_a:>10.1f}% | {max_a:>10.1f}%")
    print("-" * 75)
    print(f"Annotated verification samples saved to: {sample_output_dir.absolute()}\n")

    return stats


if __name__ == "__main__":
    audit_dataset_quality()
