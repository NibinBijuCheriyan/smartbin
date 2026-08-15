"""Quick dataset audit script — counts per-class distribution and checks label integrity."""
import os
from pathlib import Path
from collections import defaultdict

CLASSES = ["plastic", "paper", "metal", "glass", "other"]
dataset_dir = Path("data/dataset")

for split in ["train", "val"]:
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

# Check for orphan images (no label)
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

# Count TrashNet source images
print(f"\n{'='*60}")
print(f"  TRASHNET SOURCE (available but unused)")
print(f"{'='*60}")
trashnet_dir = Path("data/trashnet_extracted/dataset-resized")
if trashnet_dir.exists():
    for class_dir in sorted(trashnet_dir.iterdir()):
        if class_dir.is_dir() and class_dir.name != "__MACOSX":
            count = len([f for f in class_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
            print(f"  {class_dir.name}: {count} images")
