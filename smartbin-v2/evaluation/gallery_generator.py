"""
False Positive and False Negative Gallery Generator for SmartBin AI v2.
Extracts, visually annotates, and catalogs failure modes for qualitative debugging.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from smartbin_v2.utils.logger import get_logger

logger = get_logger("gallery_generator")


@dataclass
class FailureCase:
    case_type: str  # "false_positive" or "false_negative"
    image_name: str
    relative_path: str
    ground_truth_label: str
    predicted_label: str
    confidence: float
    iou: float


class GalleryGenerator:
    """Generates visual failure galleries with overlaid bounding boxes."""

    def __init__(self, output_dir: str | Path = "smartbin-v2/evaluation_report/galleries") -> None:
        self.output_dir = Path(output_dir)
        self.fp_dir = self.output_dir / "false_positives"
        self.fn_dir = self.output_dir / "false_negatives"

        self.fp_dir.mkdir(parents=True, exist_ok=True)
        self.fn_dir.mkdir(parents=True, exist_ok=True)

    def draw_case(
        self,
        image: np.ndarray,
        gt_box: Optional[Tuple[int, int, int, int]],
        gt_label: str,
        pred_box: Optional[Tuple[int, int, int, int]],
        pred_label: str,
        conf: float,
    ) -> np.ndarray:
        """Render ground truth (green) and predicted (red) boxes."""
        vis = image.copy()

        # Draw Ground Truth in Green
        if gt_box:
            gx1, gy1, gx2, gy2 = gt_box
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (46, 204, 113), 2)
            cv2.putText(
                vis,
                f"GT: {gt_label}",
                (gx1, max(15, gy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (46, 204, 113),
                1,
            )

        # Draw Prediction in Red
        if pred_box:
            px1, py1, px2, py2 = pred_box
            cv2.rectangle(vis, (px1, py1), (px2, py2), (231, 76, 60), 2)
            cv2.putText(
                vis,
                f"PRED: {pred_label} ({conf:.2f})",
                (px1, min(vis.shape[0] - 8, py2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (231, 76, 60),
                1,
            )

        return vis

    def save_false_positive(
        self,
        image: np.ndarray,
        image_name: str,
        pred_box: Tuple[int, int, int, int],
        pred_label: str,
        conf: float,
        gt_box: Optional[Tuple[int, int, int, int]] = None,
        gt_label: str = "none/background",
    ) -> FailureCase:
        """Annotate and catalog a False Positive error."""
        vis = self.draw_case(image, gt_box, gt_label, pred_box, pred_label, conf)
        save_path = self.fp_dir / f"fp_{image_name}"
        cv2.imwrite(str(save_path), vis)

        return FailureCase(
            case_type="false_positive",
            image_name=image_name,
            relative_path=f"galleries/false_positives/{save_path.name}",
            ground_truth_label=gt_label,
            predicted_label=pred_label,
            confidence=round(conf, 3),
            iou=0.0,
        )

    def save_false_negative(
        self,
        image: np.ndarray,
        image_name: str,
        gt_box: Tuple[int, int, int, int],
        gt_label: str,
    ) -> FailureCase:
        """Annotate and catalog a False Negative (missed object) error."""
        vis = self.draw_case(image, gt_box, gt_label, None, "missed", 0.0)
        save_path = self.fn_dir / f"fn_{image_name}"
        cv2.imwrite(str(save_path), vis)

        return FailureCase(
            case_type="false_negative",
            image_name=image_name,
            relative_path=f"galleries/false_negatives/{save_path.name}",
            ground_truth_label=gt_label,
            predicted_label="MISSED_DETECTION",
            confidence=0.0,
            iou=0.0,
        )
