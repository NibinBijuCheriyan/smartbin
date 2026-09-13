"""
Metrics Visualization Generator for SmartBin AI v2.
Produces Confusion Matrices, Precision-Recall Curves, and ROC Curves as publication-quality PNGs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from smartbin_v2.utils.logger import get_logger

logger = get_logger("metrics_generator")


class MetricsGenerator:
    """Generates visual performance diagnostics and plots."""

    def __init__(self, output_dir: str | Path = "smartbin-v2/evaluation_report") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(
        self,
        matrix: np.ndarray,
        class_names: List[str],
        filename: str = "confusion_matrix.png",
        normalize: bool = True,
    ) -> Path:
        """Render and save annotated confusion matrix heatmap."""
        out_path = self.output_dir / filename
        plt.figure(figsize=(12, 10))

        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

        plt.imshow(matrix, interpolation="nearest", cmap="Blues")
        plt.title("SmartBin AI v2 — Confusion Matrix", fontsize=15, fontweight="bold", pad=15)
        plt.colorbar(fraction=0.046, pad=0.04)

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=60, ha="right", fontsize=8)
        plt.yticks(tick_marks, class_names, fontsize=8)

        plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
        plt.ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=250)
        plt.close()

        logger.info(f"Confusion matrix plot saved: {out_path}")
        return out_path

    def plot_pr_curves(
        self,
        precisions: np.ndarray,
        recalls: np.ndarray,
        class_names: List[str],
        filename: str = "pr_curve.png",
    ) -> Path:
        """Render Precision-Recall curves across waste categories."""
        out_path = self.output_dir / filename
        plt.figure(figsize=(10, 7))

        # Sample representative curves
        for i in range(min(len(class_names), 8)):
            plt.plot(recalls, precisions[i], lw=2, label=f"{class_names[i]}")

        plt.title("Precision-Recall Curves per Waste Category", fontsize=14, fontweight="bold")
        plt.xlabel("Recall", fontsize=11)
        plt.ylabel("Precision", fontsize=11)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower left", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=250)
        plt.close()

        logger.info(f"PR curve plot saved: {out_path}")
        return out_path

    def plot_roc_curves(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        class_names: List[str],
        filename: str = "roc_curve.png",
    ) -> Path:
        """Render ROC (Receiver Operating Characteristic) curves."""
        out_path = self.output_dir / filename
        plt.figure(figsize=(10, 7))

        plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.50)")
        for i in range(min(len(class_names), 8)):
            plt.plot(fpr, tpr[i], lw=2, label=f"{class_names[i]}")

        plt.title("ROC Curves across Waste Segregation Classes", fontsize=14, fontweight="bold")
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate", fontsize=11)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=250)
        plt.close()

        logger.info(f"ROC curve plot saved: {out_path}")
        return out_path
