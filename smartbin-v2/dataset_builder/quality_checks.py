"""
Automated Dataset Quality Auditor for SmartBin AI v2.
Performs integrity checks, corrupt image detection, missing annotation audits,
bounding box spatial heatmaps, and class imbalance reports (Gini index).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from smartbin_v2.utils.logger import get_logger

logger = get_logger("dataset_quality_auditor")


def compute_gini_coefficient(frequencies: List[int]) -> float:
    """Compute Gini inequality coefficient for class frequency distribution (0=perfect equality, 1=maximum inequality)."""
    if not frequencies:
        return 0.0
    arr = np.sort(np.array(frequencies, dtype=np.float64))
    n = len(arr)
    if n == 0 or np.sum(arr) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))


class DatasetQualityAuditor:
    """Validates dataset integrity, annotations, and produces distribution heatmaps."""

    def __init__(self, class_names: Optional[Dict[int, str]] = None) -> None:
        self.class_names = class_names or {}

    def audit_dataset(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        reports_dir: str | Path = "smartbin-v2/datasets/reports",
    ) -> Dict[str, Any]:
        """
        Run exhaustive quality audit across images and labels.
        """
        img_dir = Path(images_dir)
        lbl_dir = Path(labels_dir)
        rep_dir = Path(reports_dir)
        rep_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Auditing dataset in {img_dir} and {lbl_dir}...")

        image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        label_files = list(lbl_dir.glob("*.txt"))

        img_stems = {p.stem: p for p in image_files}
        lbl_stems = {p.stem: p for p in label_files}

        corrupt_images: List[str] = []
        missing_annotations: List[str] = []
        invalid_boxes: List[Dict[str, Any]] = []
        class_counts: Counter = Counter()
        box_centers: List[Tuple[float, float]] = []

        # 1. Check images for corruption
        for stem, img_path in img_stems.items():
            try:
                with Image.open(img_path) as img:
                    img.verify()
                # Verify read with OpenCV
                test_read = cv2.imread(str(img_path))
                if test_read is None:
                    corrupt_images.append(str(img_path))
            except Exception:
                corrupt_images.append(str(img_path))

            # Check if corresponding label exists
            if stem not in lbl_stems:
                missing_annotations.append(str(img_path))

        # 2. Check labels and bounding boxes
        for stem, lbl_path in lbl_stems.items():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Check bounds
                        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                            invalid_boxes.append({
                                "file": str(lbl_path),
                                "line": line_idx,
                                "box": [cx, cy, w, h],
                                "reason": "out_of_bounds",
                            })
                        else:
                            class_counts[cls_id] += 1
                            box_centers.append((cx, cy))
                    except Exception as e:
                        invalid_boxes.append({
                            "file": str(lbl_path),
                            "line": line_idx,
                            "reason": f"syntax_error: {e}",
                        })

        # 3. Calculate statistics and class imbalance
        frequencies = list(class_counts.values())
        gini_index = compute_gini_coefficient(frequencies)
        max_freq = max(frequencies) if frequencies else 0
        min_freq = min(frequencies) if frequencies else 0
        imbalance_ratio = (max_freq / max(1, min_freq)) if min_freq > 0 else 0.0

        report = {
            "total_images": len(image_files),
            "total_labels": len(label_files),
            "total_bounding_boxes": sum(class_counts.values()),
            "corrupt_images_count": len(corrupt_images),
            "missing_annotations_count": len(missing_annotations),
            "invalid_boxes_count": len(invalid_boxes),
            "gini_inequality_index": round(gini_index, 4),
            "imbalance_ratio": round(imbalance_ratio, 2),
            "class_distribution": {
                self.class_names.get(k, f"class_{k}"): v for k, v in class_counts.items()
            },
            "corrupt_images": corrupt_images,
            "missing_annotations": missing_annotations[:50],  # Sample
            "invalid_boxes": invalid_boxes[:50],
        }

        # Save JSON Report
        report_json_path = rep_dir / "dataset_audit_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # 4. Generate Visual Charts
        self._generate_distribution_chart(class_counts, rep_dir / "class_distribution.png")
        self._generate_spatial_heatmap(box_centers, rep_dir / "bounding_box_heatmap.png")

        logger.info(f"Quality audit completed. Reports and charts saved to {rep_dir}")
        return report

    def _generate_distribution_chart(self, class_counts: Counter, output_path: Path) -> None:
        """Generate class distribution bar chart."""
        if not class_counts:
            return
        plt.figure(figsize=(12, 6))
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [self.class_names.get(k, f"Class {k}") for k, _ in sorted_classes]
        counts = [v for _, v in sorted_classes]

        plt.bar(range(len(labels)), counts, color="#2E86AB", edgecolor="black", alpha=0.85)
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=9)
        plt.title("SmartBin AI v2 — Class Instance Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Waste Classes", fontsize=11)
        plt.ylabel("Number of Bounding Boxes", fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    def _generate_spatial_heatmap(self, box_centers: List[Tuple[float, float]], output_path: Path) -> None:
        """Generate 2D spatial bounding box density heatmap."""
        if not box_centers:
            return
        xs, ys = zip(*box_centers)
        plt.figure(figsize=(7, 7))
        heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=32, range=[[0, 1], [0, 1]])
        
        plt.imshow(
            heatmap.T,
            extent=[0, 1, 1, 0],
            origin="upper",
            cmap="inferno",
            interpolation="gaussian"
        )
        plt.colorbar(label="Bounding Box Density")
        plt.title("Spatial Heatmap of Waste Objects in Frame", fontsize=13, fontweight="bold")
        plt.xlabel("Normalized X (Chute Width)", fontsize=10)
        plt.ylabel("Normalized Y (Chute Height)", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
