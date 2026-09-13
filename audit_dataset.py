"""Quick dataset audit script — counts per-class distribution, checks label integrity,
and reports intra-class visual diversity via HSV histogram variance."""
import os
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

CLASSES = ["plastic", "paper", "metal", "glass", "other"]
DATASET_DIR = Path("data/dataset")


def audit_split(dataset_dir: Path, split: str) -> None:
    """Audit a single dataset split: count classes, check label integrity."""
    labels_dir = dataset_dir / "labels" / split
    images_dir = dataset_dir / "images" / split

    class_counts = defaultdict(int)
    total_labels = 0
    malformed = []
    oob_boxes = []
    missing_images = []

    for label_file in sorted(labels_dir.glob("*.txt")):
        # Check matching image exists
        img_exists = False
        for ext in [".jpg", ".jpeg", ".png"]:
            if (images_dir / (label_file.stem + ext)).exists():
                img_exists = True
                break
        if not img_exists:
            missing_images.append(label_file.name)

        with open(label_file) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    malformed.append((label_file.name, line_no, f"expected 5 fields, got {len(parts)}"))
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    malformed.append((label_file.name, line_no, "non-numeric values"))
                    continue

                if cls_id < 0 or cls_id >= len(CLASSES):
                    malformed.append((label_file.name, line_no, f"class_id {cls_id} out of range"))
                    continue

                # Check bounding box bounds
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    oob_boxes.append((label_file.name, line_no, f"cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}"))

                # Check box doesn't extend outside image
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2
                if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
                    oob_boxes.append((label_file.name, line_no, f"box extends outside image: [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}]"))

                class_counts[cls_id] += 1
                total_labels += 1

    print(f"\n{'='*60}")
    print(f"  {split.upper()} SPLIT")
    print(f"{'='*60}")
    print(f"  Total label files: {len(list(labels_dir.glob('*.txt')))}")
    print(f"  Total image files: {len(list(images_dir.glob('*')))}")
    print(f"  Total annotations: {total_labels}")
    print()
    print(f"  {'Class ID':<10} {'Class Name':<12} {'Count':<8} {'Pct':<8}")
    print(f"  {'-'*38}")
    for idx, name in enumerate(CLASSES):
        cnt = class_counts.get(idx, 0)
        pct = (cnt / total_labels * 100) if total_labels > 0 else 0
        marker = " *** ZERO ***" if cnt == 0 else (" *** LOW ***" if cnt < 10 else "")
        print(f"  {idx:<10} {name:<12} {cnt:<8} {pct:>5.1f}%{marker}")

    if malformed:
        print(f"\n  MALFORMED LABELS ({len(malformed)}):")
        for f, ln, msg in malformed[:10]:
            print(f"    {f}:{ln} — {msg}")

    if oob_boxes:
        print(f"\n  OUT-OF-BOUNDS BOXES ({len(oob_boxes)}):")
        for f, ln, msg in oob_boxes[:10]:
            print(f"    {f}:{ln} — {msg}")

    if missing_images:
        print(f"\n  LABELS WITHOUT MATCHING IMAGE ({len(missing_images)}):")
        for f in missing_images[:10]:
            print(f"    {f}")


def audit_orphan_images(dataset_dir: Path) -> None:
    """Check for orphan images (image exists but no label)."""
    print(f"\n{'='*60}")
    print(f"  ORPHAN IMAGES (image exists but no label)")
    print(f"{'='*60}")
    for split in ["train", "val"]:
        labels_dir = dataset_dir / "labels" / split
        images_dir = dataset_dir / "images" / split
        orphans = []
        for img_file in sorted(images_dir.glob("*")):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                label_file = labels_dir / (img_file.stem + ".txt")
                if not label_file.exists():
                    orphans.append(img_file.name)
        if orphans:
            print(f"  {split}: {len(orphans)} orphan images")
            for f in orphans[:5]:
                print(f"    {f}")
        else:
            print(f"  {split}: no orphan images")


def audit_trashnet_source() -> None:
    """Count TrashNet source images."""
    print(f"\n{'='*60}")
    print(f"  TRASHNET SOURCE (available but unused)")
    print(f"{'='*60}")
    trashnet_dir = Path("data/trashnet_extracted/dataset-resized")
    if trashnet_dir.exists():
        for class_dir in sorted(trashnet_dir.iterdir()):
            if class_dir.is_dir() and class_dir.name != "__MACOSX":
                count = len([f for f in class_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
                print(f"  {class_dir.name}: {count} images")
    else:
        print("  TrashNet directory not found (data/trashnet_extracted/dataset-resized)")


def audit_class_diversity(dataset_dir: Path) -> None:
    """Compute per-class visual diversity using HSV color histogram variance.

    For each training image, computes an 8x8x8-bin HSV histogram (flattened
    to a 512-d vector). Groups histograms by class and reports:
    - Per-class image count
    - Mean intra-class histogram variance (higher = more visually diverse)
    - WARNING if "other" class count < 0.7x the mean of the 4 main classes
    """
    print(f"\n{'='*60}")
    print(f"  INTRA-CLASS VISUAL DIVERSITY (train split)")
    print(f"{'='*60}")

    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"

    if not images_dir.exists() or not labels_dir.exists():
        print("  Train split not found — skipping diversity audit.")
        return

    # Collect per-class histograms
    class_histograms: dict = {name: [] for name in CLASSES}

    for label_file in sorted(labels_dir.glob("*.txt")):
        # Find matching image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = images_dir / (label_file.stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        # Read first class ID from label (primary class for this image)
        with open(label_file) as f:
            first_line = f.readline().strip()
        if not first_line:
            continue
        parts = first_line.split()
        if len(parts) < 1:
            continue
        try:
            cls_id = int(parts[0])
        except ValueError:
            continue
        if cls_id < 0 or cls_id >= len(CLASSES):
            continue

        class_name = CLASSES[cls_id]

        # Compute 8x8x8 HSV histogram
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        class_histograms[class_name].append(hist)

    # Report per-class stats
    print()
    print(f"  {'Class':<12} {'Count':<8} {'Mean Hist Variance':<20} {'Diversity'}")
    print(f"  {'-'*55}")

    main_classes = ["plastic", "paper", "metal", "glass"]
    main_counts = []
    class_variances = {}

    for name in CLASSES:
        hists = class_histograms[name]
        count = len(hists)

        if name in main_classes:
            main_counts.append(count)

        if count >= 2:
            stacked = np.stack(hists)
            # Mean variance across histogram bins — higher = more visual spread
            mean_var = float(np.mean(np.var(stacked, axis=0)))
            class_variances[name] = mean_var
            if mean_var > 0.005:
                diversity = "HIGH"
            elif mean_var > 0.001:
                diversity = "MEDIUM"
            else:
                diversity = "LOW (tight cluster)"
            print(f"  {name:<12} {count:<8} {mean_var:<20.6f} {diversity}")
        elif count == 1:
            print(f"  {name:<12} {count:<8} {'N/A (1 sample)':<20} —")
        else:
            print(f"  {name:<12} {count:<8} {'N/A (0 samples)':<20} —")

    # Check "other" class count against main classes
    other_count = len(class_histograms["other"])
    if main_counts:
        main_avg = sum(main_counts) / len(main_counts)
        ratio = other_count / main_avg if main_avg > 0 else 0

        print()
        if ratio < 0.7:
            print(f"  WARNING: 'other' class has {other_count} images "
                  f"({ratio:.2f}x the main-class average of {main_avg:.0f}).")
            print(f"    This is below the 0.70x threshold. The 'other' class "
                  f"is undersized and likely")
            print(f"    contributes to its low precision (0.55) and recall (0.58).")
            print(f"    Consider collecting more 'other'-class training data "
                  f"or rebalancing the dataset.")
        else:
            print(f"  'other' class count ({other_count}) is {ratio:.2f}x "
                  f"the main-class average ({main_avg:.0f}) — OK.")

    # Compare diversity between "other" and main classes
    if "other" in class_variances and any(c in class_variances for c in main_classes):
        other_var = class_variances["other"]
        main_vars = [class_variances[c] for c in main_classes if c in class_variances]
        if main_vars:
            avg_main_var = sum(main_vars) / len(main_vars)
            if other_var > avg_main_var * 1.5:
                print(f"  WARNING: 'other' class has {other_var / avg_main_var:.1f}x "
                      f"the visual variance of main classes.")
                print(f"    This suggests 'other' is visually incoherent (catch-all of "
                      f"batteries, food waste, shoes, etc.).")
                print(f"    Consider splitting 'other' into sub-categories or using "
                      f"a higher confidence threshold.")


if __name__ == "__main__":
    for split in ["train", "val"]:
        audit_split(DATASET_DIR, split)

    audit_orphan_images(DATASET_DIR)
    audit_trashnet_source()
    audit_class_diversity(DATASET_DIR)

