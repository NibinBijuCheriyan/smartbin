"""
Hard Negative Mining and Distractor Suppression Pipeline for SmartBin AI v2.
Integrates non-waste objects (hands, mobile phones, wallets, keys, shoes, clothes, empty tables)
as background images with zero annotations, training the model to suppress false alarms.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from smartbin_v2.utils.logger import get_logger

logger = get_logger("hard_negative_miner")

# Non-waste distractor categories
DISTRACTOR_CATEGORIES = [
    "human_hand",
    "mobile_phone",
    "wallet",
    "keys",
    "shoes",
    "clothes",
    "empty_chute",
    "floor_surface",
]


class HardNegativeMiner:
    """
    Manages non-waste background images to teach the model to ignore non-trash objects.
    Ultralytics YOLO treats an image with an empty .txt label file as a background sample.
    """

    def __init__(
        self,
        negatives_dir: str | Path = "smartbin-v2/datasets/hard_negatives",
        unknown_rejection_thresh: float = 0.40,
    ) -> None:
        self.negatives_dir = Path(negatives_dir)
        self.negatives_dir.mkdir(parents=True, exist_ok=True)
        self.rejection_thresh = unknown_rejection_thresh

    def ingest_distractor_image(
        self,
        image_path: str | Path,
        category: str,
        target_splits_dir: str | Path = "smartbin-v2/datasets/splits/train",
    ) -> bool:
        """
        Copy distractor image into training split and create an empty .txt label file.
        """
        src = Path(image_path)
        if not src.exists():
            return False

        split_img_dir = Path(target_splits_dir) / "images"
        split_lbl_dir = Path(target_splits_dir) / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        dest_img_name = f"bg_{category}_{src.name}"
        dest_img = split_img_dir / dest_img_name
        dest_lbl = split_lbl_dir / f"{Path(dest_img_name).stem}.txt"

        # Copy image
        shutil.copy2(src, dest_img)

        # Create completely empty label file (background image in YOLO)
        dest_lbl.touch()

        logger.info(f"Ingested hard negative background: {dest_img_name} ({category})")
        return True

    def evaluate_unknown_waste_rejection(
        self,
        confidence: float,
        detected_class: str,
    ) -> Tuple[bool, str]:
        """
        Enforce unknown waste rejection:
        If confidence < rejection_thresh or class matches distractor, reject waste item.
        Returns: (is_rejected, rejection_reason)
        """
        if confidence < self.rejection_thresh:
            return True, f"confidence_too_low_{confidence:.2f}"

        if detected_class in DISTRACTOR_CATEGORIES or detected_class == "unknown_waste":
            return True, f"distractor_or_unknown_class_{detected_class}"

        return False, "accepted"
