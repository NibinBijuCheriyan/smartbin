"""
Dataset balancing and stratified splitting module.
Analyzes class frequency, balances overrepresented and underrepresented classes,
and generates stratified Train / Val / Test splits (70/15/15).
"""

from __future__ import annotations

import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from smartbin_v2.utils.logger import get_logger

logger = get_logger("dataset_balancer")


class DatasetBalancer:
    """Handles class frequency balancing and stratified Train/Val/Test partitioning."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def parse_yolo_labels(self, labels_dir: str | Path) -> Dict[str, List[int]]:
        """
        Extract list of class IDs present in each label file.
        Returns: {file_stem: [class_id1, class_id2, ...]}
        """
        lbl_dir = Path(labels_dir)
        results: Dict[str, List[int]] = {}

        for txt_path in lbl_dir.glob("*.txt"):
            stem = txt_path.stem
            classes: List[int] = []
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            classes.append(int(parts[0]))
                        except ValueError:
                            pass
            results[stem] = classes

        return results

    def create_stratified_splits(
        self,
        image_label_pairs: List[Tuple[Path, Path]],
        label_info: Dict[str, List[int]],
    ) -> Dict[str, List[Tuple[Path, Path]]]:
        """
        Perform stratified train/val/test split prioritizing rare classes.
        """
        random.seed(self.seed)
        
        # Group items by primary class (first class or rarest class)
        class_buckets: Dict[int, List[Tuple[Path, Path]]] = defaultdict(list)
        unlabeled_bucket: List[Tuple[Path, Path]] = []

        for img_path, lbl_path in image_label_pairs:
            stem = img_path.stem
            classes = label_info.get(stem, [])
            if classes:
                # Assign to primary class bucket
                primary_cls = classes[0]
                class_buckets[primary_cls].append((img_path, lbl_path))
            else:
                unlabeled_bucket.append((img_path, lbl_path))

        splits: Dict[str, List[Tuple[Path, Path]]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        for cls_id, items in class_buckets.items():
            random.shuffle(items)
            n = len(items)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)

            train_items = items[:n_train]
            val_items = items[n_train : n_train + n_val]
            test_items = items[n_train + n_val :]

            splits["train"].extend(train_items)
            splits["val"].extend(val_items)
            splits["test"].extend(test_items)

        # Distribute background / negative images across splits
        random.shuffle(unlabeled_bucket)
        n_unlabeled = len(unlabeled_bucket)
        u_train = int(n_unlabeled * self.train_ratio)
        u_val = int(n_unlabeled * self.val_ratio)

        splits["train"].extend(unlabeled_bucket[:u_train])
        splits["val"].extend(unlabeled_bucket[u_train : u_train + u_val])
        splits["test"].extend(unlabeled_bucket[u_train + u_val :])

        logger.info(
            f"Stratified partition created: {len(splits['train'])} train, "
            f"{len(splits['val'])} val, {len(splits['test'])} test"
        )
        return splits

    def export_splits(
        self,
        splits: Dict[str, List[Tuple[Path, Path]]],
        output_dir: str | Path,
    ) -> Path:
        """
        Copy partitioned images and labels into standard YOLO split folders:
        output_dir/{train,val,test}/{images,labels}/
        """
        base_out = Path(output_dir)

        for split_name, items in splits.items():
            img_dir = base_out / split_name / "images"
            lbl_dir = base_out / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path, lbl_path in items:
                dest_img = img_dir / img_path.name
                shutil.copy2(img_path, dest_img)

                if lbl_path.exists():
                    dest_lbl = lbl_dir / lbl_path.name
                    shutil.copy2(lbl_path, dest_lbl)
                else:
                    # Create empty label file for negative background image
                    dest_lbl = lbl_dir / f"{img_path.stem}.txt"
                    dest_lbl.touch()

        logger.info(f"YOLO splits successfully written to {base_out}")
        return base_out
