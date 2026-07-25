"""
Cashcrow Smartbin — Training Pipeline.

Downloads TrashNet and TACO datasets, remaps classes to the 7 Cashcrow waste
categories, and fine-tunes a YOLO model for waste detection.

Key design decisions:
- TrashNet images use GrabCut foreground segmentation for bounding boxes
  (not full-frame labels, which train unrealistic box geometry).
- Real-data failures raise loud exceptions — mock data is ONLY used with --mock.
- Dataset integrity is checked before training starts (min images per class).
- Domain gap is explicitly warned: TrashNet/TACO alone are insufficient for
  deployment without first-party Cashcrow bin images.

Usage:
    # Train on real data (downloads TrashNet + TACO):
    python train_waste_model.py

    # Train with synthetic mock data (for testing the pipeline):
    python train_waste_model.py --mock

    # Include first-party Cashcrow bin images:
    python train_waste_model.py --cashcrow-data path/to/labeled/images

    # Allow partial data (e.g., only TrashNet succeeded):
    python train_waste_model.py --allow-partial
"""

import os
import sys
import zipfile
import json
import random
import argparse
import shutil
import urllib.request
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class mappings
# ---------------------------------------------------------------------------

# Mapping of TACO category names to Cashcrow target classes
TACO_TO_CASHCROW = {
    "Aluminium foil": "metal",
    "Battery": "e-waste",
    "Aluminium blister pack": "metal",
    "Carded blister pack": "paper",
    "Clear plastic bottle": "plastic",
    "Glass bottle": "glass",
    "Other plastic bottle": "plastic",
    "Plastic bottle cap": "plastic",
    "Metal bottle cap": "metal",
    "Broken glass": "glass",
    "Drink can": "metal",
    "Food Can": "metal",
    "Food can": "metal",
    "Corrugated carton": "paper",
    "Drink carton": "paper",
    "Egg carton": "paper",
    "Meal carton": "paper",
    "Other carton": "paper",
    "Paper cup": "paper",
    "Disposable plastic cup": "plastic",
    "Foam cup": "plastic",
    "Glass cup": "glass",
    "Other plastic cup": "plastic",
    "Food waste": "organic",
    "Plastic lid": "plastic",
    "Metal lid": "metal",
    "Magazine paper": "paper",
    "Wrapping paper": "paper",
    "Pizza box": "paper",
    "Paper bag": "paper",
    "Plastic bag - wrapper": "plastic",
    "Cigarette": "other",
    "Cigarette box": "other",
    "Unlabeled litter": "other",
    "Garbage bag": "plastic",
    "Single-use carrier bag": "plastic",
    "Polyethylene bag": "plastic",
    "Straw": "other",
    "Paper straw": "paper",
    "Plastic straw": "plastic",
    "Plastic utensils": "plastic",
    "Plastic glooves": "plastic",
    "Plastic gloves": "plastic",
    "Paper glooves": "paper",
    "Paper gloves": "paper",
    "Metal glooves": "metal",
    "Metal gloves": "metal",
    "Plastic film": "plastic",
    "Squeezable tube": "plastic",
    "Toothbrush": "other",
    "Shoe": "other",
    "Polypropylene bag": "plastic",
    "Toilet paper": "paper",
    "Plaster": "other",
    "Glass jar": "glass",
    "Foil paper": "metal",
    "Foil wrapper": "metal",
    "Bubble wrap": "plastic",
    "Plastic bottle pack": "plastic",
    "Carded pack": "paper",
    "Cardboard box": "paper",
    "Other paper": "paper",
    "Tetra pack": "paper",
}

# Mapping of TrashNet categories to Cashcrow target classes
TRASHNET_TO_CASHCROW = {
    "glass": "glass",
    "paper": "paper",
    "cardboard": "paper",
    "plastic": "plastic",
    "metal": "metal",
    "trash": "other",
}

CLASSES = ["plastic", "paper", "metal", "glass", "e-waste", "organic", "other"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

# Minimum images per class in the training split to consider the dataset usable
MIN_IMAGES_PER_CLASS = 20


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatasetError(Exception):
    """Raised when dataset preparation fails."""


class DatasetIntegrityError(Exception):
    """Raised when the dataset fails integrity checks before training."""


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_file(url: str, output_path: Path) -> None:
    """Downloads a file with basic progress indication."""
    logger.info("Downloading %s to %s...", url, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response, open(
            output_path, "wb"
        ) as out_file:
            shutil.copyfileobj(response, out_file)
    except Exception as e:
        # Clean up partial downloads
        if output_path.exists():
            output_path.unlink()
        raise DatasetError(f"Failed to download {url}: {e}") from e

    logger.info("Download completed: %s", output_path)


# ---------------------------------------------------------------------------
# GrabCut foreground segmentation for TrashNet images
# ---------------------------------------------------------------------------


def extract_foreground_bbox(
    img: np.ndarray, margin_fraction: float = 0.05
) -> tuple:
    """
    Use GrabCut to segment the foreground object and return its bounding box
    in YOLO normalized format (cx, cy, w, h).

    TrashNet images are studio shots with one item against a plain background,
    which is ideal for GrabCut. If GrabCut fails to find a clear foreground,
    falls back to a padded center crop (70% of frame).

    Args:
        img: BGR image (numpy array).
        margin_fraction: Margin around the image edges for GrabCut init rect.

    Returns:
        Tuple (cx, cy, w, h) in normalized [0, 1] coordinates.
    """
    h, w = img.shape[:2]

    # Initial rectangle for GrabCut (exclude a small margin around edges)
    margin_x = max(5, int(w * margin_fraction))
    margin_y = max(5, int(h * margin_fraction))
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)

        # Foreground = definite foreground (3) + probable foreground (1)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(
            np.uint8
        )

        # Find bounding box of the foreground region
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Use the largest contour
            largest = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest)

            # Only accept if the foreground is a reasonable size (>5% of image)
            area_fraction = (bw * bh) / (w * h)
            if area_fraction > 0.05:
                cx = (x + bw / 2.0) / w
                cy = (y + bh / 2.0) / h
                nw = bw / w
                nh = bh / h
                return (cx, cy, nw, nh)
    except cv2.error:
        pass  # GrabCut can fail on very small or uniform images

    # Fallback: padded center crop (70% of frame)
    logger.debug("GrabCut fallback: using 70%% center crop")
    return (0.5, 0.5, 0.7, 0.7)


# ---------------------------------------------------------------------------
# Mock (synthetic) dataset generation
# ---------------------------------------------------------------------------


def generate_mock_dataset(dataset_dir: Path, num_images: int = 140) -> None:
    """Generates a synthetic dataset for testing the training and benchmark flow."""
    logger.info("Generating synthetic mock dataset...")

    # Create directory structure
    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    splits = ["train"] * int(num_images * 0.8) + ["val"] * (
        num_images - int(num_images * 0.8)
    )
    random.shuffle(splits)

    # Visual features of each class to make synthetic images slightly realistic
    class_visuals = {
        "plastic": {"color": (255, 0, 0), "shape": "rect"},
        "paper": {"color": (19, 136, 219), "shape": "rect"},
        "metal": {"color": (192, 192, 192), "shape": "circle"},
        "glass": {"color": (0, 255, 0), "shape": "ellipse"},
        "e-waste": {"color": (0, 0, 255), "shape": "poly"},
        "organic": {"color": (0, 255, 255), "shape": "star"},
        "other": {"color": (128, 128, 128), "shape": "line"},
    }

    for i, split in enumerate(splits):
        class_name = CLASSES[i % len(CLASSES)]
        class_idx = CLASS_TO_IDX[class_name]

        # Create a blank image with varied background
        bg_val = random.randint(30, 80)
        img = np.full((320, 320, 3), (bg_val, bg_val, bg_val), dtype=np.uint8)

        # Add noise
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw a synthetic object
        vis = class_visuals[class_name]
        cx = random.randint(100, 220)
        cy = random.randint(100, 220)
        w = random.randint(60, 140)
        h = random.randint(60, 140)

        x1 = max(10, cx - w // 2)
        y1 = max(10, cy - h // 2)
        x2 = min(310, cx + w // 2)
        y2 = min(310, cy + h // 2)

        # Draw on image
        color = vis["color"]
        if vis["shape"] == "rect":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        elif vis["shape"] == "circle":
            cv2.circle(img, (cx, cy), w // 2, color, -1)
        elif vis["shape"] == "ellipse":
            cv2.ellipse(
                img,
                (cx, cy),
                (w // 2, h // 2),
                random.randint(0, 360),
                0,
                360,
                color,
                -1,
            )
        else:
            pts = np.array(
                [[cx, y1], [x2, cy], [cx, y2], [x1, cy]], np.int32
            )
            cv2.fillPoly(img, [pts], color)

        # Save image
        img_name = f"{class_name}_{i:04d}.jpg"
        img_path = dataset_dir / "images" / split / img_name
        cv2.imwrite(str(img_path), img)

        # Normalize bounding box for YOLO
        nx = cx / 320.0
        ny = cy / 320.0
        nw = (x2 - x1) / 320.0
        nh = (y2 - y1) / 320.0

        # Save label
        label_path = dataset_dir / "labels" / split / f"{class_name}_{i:04d}.txt"
        with open(label_path, "w") as f:
            f.write(f"{class_idx} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")

    logger.info("Mock dataset generated at %s (%d images)", dataset_dir, num_images)


# ---------------------------------------------------------------------------
# TrashNet download and processing
# ---------------------------------------------------------------------------


def download_trashnet(data_dir: Path) -> Path:
    """
    Download and extract the TrashNet dataset.

    Returns:
        Path to the extracted TrashNet root directory.

    Raises:
        DatasetError: If download or extraction fails.
    """
    trashnet_zip = data_dir / "trashnet.zip"
    trashnet_extract_dir = data_dir / "trashnet_extracted"

    if not trashnet_zip.exists():
        download_file(
            "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip",
            trashnet_zip,
        )

    if not trashnet_extract_dir.exists():
        logger.info("Extracting TrashNet...")
        try:
            with zipfile.ZipFile(trashnet_zip, "r") as zip_ref:
                zip_ref.extractall(trashnet_extract_dir)
        except (zipfile.BadZipFile, OSError) as e:
            raise DatasetError(
                f"Failed to extract TrashNet archive ({trashnet_zip}): {e}"
            ) from e

    return trashnet_extract_dir


def process_trashnet(
    trashnet_dir: Path,
    dataset_dir: Path,
    trashnet_mode: str = "grabcut",
    max_images: int = 500,
) -> int:
    """
    Process TrashNet images into YOLO detection format.

    Args:
        trashnet_dir: Path to extracted TrashNet root.
        dataset_dir: Target dataset directory.
        trashnet_mode: How to handle TrashNet images:
            - "grabcut": Use GrabCut foreground segmentation for bounding boxes.
            - "drop": Skip TrashNet entirely (rely on TACO only).
        max_images: Maximum number of TrashNet images to include.

    Returns:
        Number of images processed.
    """
    if trashnet_mode == "drop":
        logger.info("TrashNet mode='drop': skipping TrashNet images.")
        return 0

    logger.info("Processing TrashNet images (mode=%s)...", trashnet_mode)
    trashnet_orig_dir = trashnet_dir / "dataset-resized"

    if not trashnet_orig_dir.exists():
        raise DatasetError(
            f"TrashNet extracted directory not found: {trashnet_orig_dir}. "
            f"Archive may be corrupted — delete {trashnet_dir} and retry."
        )

    all_images = []
    for class_dir_name in os.listdir(trashnet_orig_dir):
        class_path = trashnet_orig_dir / class_dir_name
        if not class_path.is_dir():
            continue

        mapped_class = TRASHNET_TO_CASHCROW.get(class_dir_name)
        if not mapped_class:
            continue

        class_idx = CLASS_TO_IDX[mapped_class]

        for file_name in os.listdir(class_path):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append((class_path / file_name, class_idx, mapped_class))

    random.shuffle(all_images)
    subset = all_images[:max_images]

    count = 0
    for idx, (img_path, class_idx, mapped_class) in enumerate(subset):
        split = "train" if idx < len(subset) * 0.8 else "val"
        dest_img = dataset_dir / "images" / split / f"trashnet_{idx:04d}.jpg"

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read TrashNet image: %s", img_path)
            continue

        shutil.copy(img_path, dest_img)

        # Compute bounding box via GrabCut foreground segmentation
        cx, cy, nw, nh = extract_foreground_bbox(img)

        dest_label = dataset_dir / "labels" / split / f"trashnet_{idx:04d}.txt"
        with open(dest_label, "w") as lf:
            lf.write(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        count += 1

    logger.info(
        "Processed %d TrashNet images (mode=%s)", count, trashnet_mode
    )
    return count


# ---------------------------------------------------------------------------
# TACO download and processing
# ---------------------------------------------------------------------------


def download_taco(data_dir: Path) -> Path:
    """
    Download TACO annotations JSON.

    Returns:
        Path to the annotations JSON file.

    Raises:
        DatasetError: If download fails.
    """
    taco_json = data_dir / "taco_annotations.json"

    if not taco_json.exists():
        download_file(
            "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json",
            taco_json,
        )

    return taco_json


def process_taco(
    taco_json: Path,
    dataset_dir: Path,
    max_images: int = 100,
) -> int:
    """
    Download and process TACO images + annotations into YOLO format.

    TACO provides real bounding box annotations, so no GrabCut is needed.

    Returns:
        Number of images processed.
    """
    logger.info("Processing TACO annotations from %s...", taco_json)

    try:
        with open(taco_json, "r") as f:
            taco_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise DatasetError(
            f"Failed to parse TACO annotations ({taco_json}): {e}"
        ) from e

    taco_categories = {cat["id"]: cat["name"] for cat in taco_data["categories"]}

    # Group annotations by image
    img_annotations: dict = {}
    for ann in taco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    taco_images = taco_data["images"]
    random.shuffle(taco_images)

    taco_count = 0
    for img_entry in taco_images:
        if taco_count >= max_images:
            break

        img_id = img_entry["id"]
        if img_id not in img_annotations:
            continue

        url = img_entry.get("flickr_640_url") or img_entry.get("flickr_url")
        if not url:
            continue

        # Remap classes for this image
        labels_lines = []
        img_w = img_entry["width"]
        img_h = img_entry["height"]

        for ann in img_annotations[img_id]:
            cat_name = taco_categories.get(ann["category_id"])
            mapped_class = TACO_TO_CASHCROW.get(cat_name)
            if not mapped_class:
                continue
            class_idx = CLASS_TO_IDX[mapped_class]

            # Bbox is [x, y, width, height] in absolute coordinates
            bbox = ann["bbox"]
            bx = (bbox[0] + bbox[2] / 2) / img_w
            by = (bbox[1] + bbox[3] / 2) / img_h
            bw = bbox[2] / img_w
            bh = bbox[3] / img_h
            labels_lines.append(
                f"{class_idx} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}"
            )

        if not labels_lines:
            continue

        split = "train" if taco_count < max_images * 0.8 else "val"
        dest_img = dataset_dir / "images" / split / f"taco_{taco_count:04d}.jpg"

        try:
            download_file(url, dest_img)
            dest_label = (
                dataset_dir / "labels" / split / f"taco_{taco_count:04d}.txt"
            )
            with open(dest_label, "w") as lf:
                lf.write("\n".join(labels_lines) + "\n")
            taco_count += 1
        except DatasetError as e:
            logger.warning("Skipping TACO image %d: %s", img_id, e)

    logger.info("Processed %d TACO images.", taco_count)
    return taco_count


# ---------------------------------------------------------------------------
# First-party Cashcrow data integration
# ---------------------------------------------------------------------------


def integrate_cashcrow_data(cashcrow_dir: Path, dataset_dir: Path) -> int:
    """
    Copy first-party Cashcrow bin images + labels into the dataset.

    Expects YOLO format: cashcrow_dir/images/{train,val}/*.jpg
                         cashcrow_dir/labels/{train,val}/*.txt

    Returns:
        Number of images integrated.
    """
    count = 0
    for split in ["train", "val"]:
        src_images = cashcrow_dir / "images" / split
        src_labels = cashcrow_dir / "labels" / split

        if not src_images.exists():
            logger.warning(
                "Cashcrow data split '%s' not found at %s", split, src_images
            )
            continue

        dst_images = dataset_dir / "images" / split
        dst_labels = dataset_dir / "labels" / split

        for img_file in src_images.iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                dest_img = dst_images / f"cashcrow_{img_file.name}"
                shutil.copy(img_file, dest_img)

                # Copy corresponding label
                label_name = img_file.stem + ".txt"
                src_label = src_labels / label_name
                if src_label.exists():
                    shutil.copy(src_label, dst_labels / f"cashcrow_{label_name}")
                else:
                    logger.warning(
                        "Missing label for Cashcrow image: %s", img_file.name
                    )

                count += 1

    logger.info("Integrated %d first-party Cashcrow images.", count)
    return count


# ---------------------------------------------------------------------------
# Dataset integrity check
# ---------------------------------------------------------------------------


def check_dataset_integrity(
    dataset_dir: Path, min_per_class: int = MIN_IMAGES_PER_CLASS, allow_sparse: bool = False
) -> None:
    """
    Validate dataset before training: check each class has enough images.

    Logs a class-by-class breakdown and raises if any class is under-represented.
    """
    logger.info("=" * 60)
    logger.info("Dataset Integrity Check")
    logger.info("=" * 60)

    class_counts: dict = {
        "train": defaultdict(int),
        "val": defaultdict(int),
    }

    for split in ["train", "val"]:
        labels_dir = dataset_dir / "labels" / split
        if not labels_dir.exists():
            raise DatasetIntegrityError(
                f"Labels directory not found: {labels_dir}"
            )

        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_idx = int(parts[0])
                        if 0 <= class_idx < len(CLASSES):
                            class_counts[split][CLASSES[class_idx]] += 1

    # Log breakdown
    logger.info("")
    logger.info("%-12s | %10s | %10s", "Class", "Train", "Val")
    logger.info("-" * 40)

    total_train = 0
    total_val = 0
    starved_classes = []

    for cls in CLASSES:
        train_n = class_counts["train"].get(cls, 0)
        val_n = class_counts["val"].get(cls, 0)
        total_train += train_n
        total_val += val_n

        marker = ""
        if train_n < min_per_class:
            marker = " *** UNDER-REPRESENTED ***"
            starved_classes.append((cls, train_n))

        logger.info("%-12s | %10d | %10d%s", cls, train_n, val_n, marker)

    logger.info("-" * 40)
    logger.info("%-12s | %10d | %10d", "TOTAL", total_train, total_val)
    logger.info("")

    if starved_classes:
        msg = (
            f"Dataset integrity check FAILED. The following classes have fewer "
            f"than {min_per_class} training images:\n"
        )
        for cls, count in starved_classes:
            msg += f"  - {cls}: {count} images\n"
        msg += (
            "Add more images for these classes or use --allow-sparse to proceed anyway."
        )

        if allow_sparse:
            logger.warning("SPARSE DATASET: Proceeding despite under-represented classes (--allow-sparse).")
            logger.warning(msg)
        else:
            raise DatasetIntegrityError(msg)

    logger.info("Dataset integrity check PASSED.")


# ---------------------------------------------------------------------------
# Real dataset setup (orchestrator)
# ---------------------------------------------------------------------------


def setup_real_dataset(
    data_dir: Path,
    allow_partial: bool = False,
    trashnet_mode: str = "grabcut",
    cashcrow_data: str = None,
) -> Path:
    """
    Download and construct the real training dataset from TrashNet and TACO.

    This function NEVER falls back to mock data silently. On failure, it raises
    a clear exception naming which step failed.

    Args:
        data_dir: Base data directory.
        allow_partial: If True, allow training with only one source dataset.
        trashnet_mode: How to handle TrashNet images ("grabcut" or "drop").
        cashcrow_data: Optional path to first-party Cashcrow labeled images.

    Returns:
        Path to the constructed dataset directory.

    Raises:
        DatasetError: If a download or processing step fails.
    """
    dataset_dir = data_dir / "dataset"

    # Create directory structure
    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Track which sources succeeded
    trashnet_ok = False
    taco_ok = False
    trashnet_count = 0
    taco_count = 0

    # --- TrashNet ---
    if trashnet_mode != "drop":
        try:
            trashnet_dir = download_trashnet(data_dir)
            trashnet_count = process_trashnet(
                trashnet_dir, dataset_dir, trashnet_mode=trashnet_mode
            )
            trashnet_ok = True
        except DatasetError as e:
            logger.error("TrashNet preparation FAILED: %s", e)
            if not allow_partial:
                raise DatasetError(
                    f"TrashNet download/processing failed: {e}. "
                    f"Use --allow-partial to continue with TACO only, "
                    f"or --mock for synthetic data."
                ) from e
            logger.warning("Continuing without TrashNet (--allow-partial).")
    else:
        logger.info("TrashNet skipped (--trashnet-mode=drop).")

    # --- TACO ---
    try:
        taco_json = download_taco(data_dir)
        taco_count = process_taco(taco_json, dataset_dir)
        taco_ok = True
    except DatasetError as e:
        logger.error("TACO preparation FAILED: %s", e)
        if not allow_partial:
            raise DatasetError(
                f"TACO download/processing failed: {e}. "
                f"Use --allow-partial to continue with TrashNet only, "
                f"or --mock for synthetic data."
            ) from e
        logger.warning("Continuing without TACO (--allow-partial).")

    # --- First-party Cashcrow data ---
    cashcrow_count = 0
    if cashcrow_data:
        cashcrow_path = Path(cashcrow_data)
        if not cashcrow_path.exists():
            raise DatasetError(
                f"Cashcrow data directory not found: {cashcrow_path}"
            )
        cashcrow_count = integrate_cashcrow_data(cashcrow_path, dataset_dir)
    else:
        logger.warning("")
        logger.warning("=" * 70)
        logger.warning(
            "WARNING: Training without first-party Cashcrow bin images."
        )
        logger.warning(
            "TrashNet (studio shots) and TACO (outdoor litter) do NOT"
        )
        logger.warning(
            "resemble the actual deployment scene (hand holding item"
        )
        logger.warning(
            "over a bin, indoor lighting, close range)."
        )
        logger.warning(
            "Use --cashcrow-data to include target-domain images."
        )
        logger.warning("=" * 70)
        logger.warning("")

    # Check we got at least some data
    total = trashnet_count + taco_count + cashcrow_count
    if total == 0:
        raise DatasetError(
            "No images were successfully processed from any source. "
            "Check network connectivity and try again, or use --mock."
        )

    if not trashnet_ok and not taco_ok and cashcrow_count == 0:
        raise DatasetError(
            "Both TrashNet and TACO failed, and no Cashcrow data was provided. "
            "Cannot proceed with training."
        )

    logger.info("")
    logger.info("Dataset summary: TrashNet=%d, TACO=%d, Cashcrow=%d, Total=%d",
                trashnet_count, taco_count, cashcrow_count, total)

    return dataset_dir


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------


def detect_device(requested: str = "auto") -> str:
    """
    Auto-detect the best available device for training.

    Priority: user override > CUDA > MPS > CPU.
    """
    if requested != "auto":
        logger.info("Using explicitly requested device: %s", requested)
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda:0"
            logger.info("CUDA GPU detected: %s", torch.cuda.get_device_name(0))
            return device
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS device detected.")
            return "mps"
    except ImportError:
        pass

    logger.info("No GPU detected — using CPU.")
    return "cpu"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a YOLO waste detection model for the Cashcrow Smartbin."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic mock dataset instead of real data (for testing only).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience — stop if no improvement for N epochs (default: 10).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: 'auto' (detect), 'cpu', 'cuda:0', 'mps' (default: auto).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Continue training if only one data source (TrashNet or TACO) succeeds.",
    )
    parser.add_argument(
        "--allow-sparse",
        action="store_true",
        help="Allow training even if some classes have fewer than the minimum required images.",
    )
    parser.add_argument(
        "--trashnet-mode",
        type=str,
        default="grabcut",
        choices=["grabcut", "drop"],
        help=(
            "How to handle TrashNet images: "
            "'grabcut' = GrabCut segmentation for bounding boxes (default), "
            "'drop' = exclude TrashNet entirely."
        ),
    )
    parser.add_argument(
        "--cashcrow-data",
        type=str,
        default=None,
        help="Path to first-party Cashcrow bin images (YOLO format: images/{train,val}, labels/{train,val}).",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    dataset_dir = data_dir / "dataset"

    # Clean previous dataset if any
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    # --- Build dataset ---
    if args.mock:
        logger.info("Using synthetic mock dataset (--mock flag).")
        generate_mock_dataset(dataset_dir)
    else:
        setup_real_dataset(
            data_dir,
            allow_partial=args.allow_partial,
            trashnet_mode=args.trashnet_mode,
            cashcrow_data=args.cashcrow_data,
        )

    # --- Dataset integrity check ---
    check_dataset_integrity(
        dataset_dir,
        min_per_class=MIN_IMAGES_PER_CLASS if not args.mock else 1,
        allow_sparse=args.allow_sparse or args.mock,
    )

    # --- Create dataset.yaml ---
    dataset_yaml = data_dir / "dataset.yaml"
    yaml_content = f"""path: {dataset_dir.absolute().as_posix()}
train: images/train
val: images/val

names:
  0: plastic
  1: paper
  2: metal
  3: glass
  4: e-waste
  5: organic
  6: other
"""
    with open(dataset_yaml, "w") as f:
        f.write(yaml_content)

    logger.info("Dataset configuration written to %s", dataset_yaml)

    # --- Detect device ---
    device = detect_device(args.device)

    # --- Train YOLO Model ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("Starting YOLO training")
    logger.info("  Epochs:   %d (patience=%d)", args.epochs, args.patience)
    logger.info("  Image sz: %d", args.imgsz)
    logger.info("  Device:   %s", device)
    logger.info("=" * 60)

    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")

    # Run training with augmentations appropriate for small datasets
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        device=device,
        # Data augmentation for small datasets
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        workers=2,
    )

    logger.info("Augmentation settings: degrees=15, translate=0.1, scale=0.5, "
                "hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, mosaic=1.0, mixup=0.1, copy_paste=0.1")

    # --- Copy best weights ---
    best_weights_dst = Path("best.pt")

    search_dirs = [
        Path("runs/detect"),
        Path.home() / "runs" / "detect",
    ]

    found_weights = None
    for sdir in search_dirs:
        if sdir.exists():
            train_folders = sorted(
                list(sdir.glob("train*")),
                key=os.path.getmtime,
                reverse=True,
            )
            for tf in train_folders:
                candidate = tf / "weights" / "best.pt"
                if candidate.exists():
                    found_weights = candidate
                    break
        if found_weights:
            break

    if found_weights and found_weights.exists():
        shutil.copy(found_weights, best_weights_dst)
        logger.info(
            "Fine-tuned weights copied to %s from %s",
            best_weights_dst.absolute(),
            found_weights,
        )
    else:
        logger.error("Could not locate best.pt weights file after training.")


if __name__ == "__main__":
    main()
