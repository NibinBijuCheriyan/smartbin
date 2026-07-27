"""
Evaluate the Cashcrow EfficientNet-B0 Waste Classifier on Ground-Truth Crops.

Loads labeled waste crops (or samples from real datasets / video detections),
runs inference using WasteClassifier (TFLite FP32), and computes true measured
accuracy, recall, precision, and F1-score.

Usage:
    python evaluate_smartbin_crops.py [--dataset-dir PATH]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from smartbin.config import RefinerConfig
from smartbin.refiner import WasteClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Category mapping from dataset folders to EfficientNet-B0 target 5 classes:
# plastic, paper, metal, organic_waste, none
DATASET_CLASS_MAP = {
    "plastic": "plastic",
    "paper": "paper",
    "cardboard": "paper",
    "metal": "metal",
    "trash": "organic_waste",
    "organic": "organic_waste",
    "none": "none",
}


def load_ground_truth_crops(dataset_dir: Path, samples_per_class: int = 20) -> List[Tuple[np.ndarray, str]]:
    """Load images with known ground-truth class labels from dataset subdirectories."""
    crops_and_labels: List[Tuple[np.ndarray, str]] = []

    if not dataset_dir.exists():
        logger.warning("Dataset directory not found: %s", dataset_dir)
        return crops_and_labels

    for folder_name, target_class in DATASET_CLASS_MAP.items():
        class_folder = dataset_dir / folder_name
        if not class_folder.is_dir():
            continue

        image_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
        sampled_files = image_files[:samples_per_class]

        for img_path in sampled_files:
            crop = cv2.imread(str(img_path))
            if crop is not None and crop.size > 0:
                crops_and_labels.append((crop, target_class))

    logger.info("Loaded %d ground-truth crop samples across dataset categories.", len(crops_and_labels))
    return crops_and_labels


def evaluate_classifier(
    crops_and_labels: List[Tuple[np.ndarray, str]],
    model_path: str,
    classes_path: str,
    confidence_threshold: float = 0.25,
) -> Dict[str, float]:
    """Run WasteClassifier inference on crops and compute accuracy, recall, precision, and F1."""
    classifier = WasteClassifier(
        model_path=model_path,
        classes_path=classes_path,
        confidence_threshold=confidence_threshold,
    )

    correct = 0
    total = len(crops_and_labels)

    class_stats: Dict[str, Dict[str, int]] = {}

    for crop, true_label in crops_and_labels:
        pred_label, conf, _ = classifier.classify(crop)

        if true_label not in class_stats:
            class_stats[true_label] = {"tp": 0, "fp": 0, "fn": 0, "total": 0}
        if pred_label not in class_stats:
            class_stats[pred_label] = {"tp": 0, "fp": 0, "fn": 0, "total": 0}

        class_stats[true_label]["total"] += 1

        if pred_label == true_label:
            correct += 1
            class_stats[true_label]["tp"] += 1
        else:
            class_stats[true_label]["fn"] += 1
            class_stats[pred_label]["fp"] += 1

    accuracy = (correct / total * 100.0) if total > 0 else 0.0

    print("\n==========================================================================")
    print("           SMARTBIN REFINER GROUND-TRUTH CROP EVALUATION REPORT           ")
    print("==========================================================================")
    print(f"Total Evaluated Crops : {total}")
    print(f"Top-1 Accuracy        : {accuracy:.2f}%\n")

    print(f"{'Class Name':<15} | {'Count':<7} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)

    recalls, precisions, f1s = [], [], []

    for cls_name, stats in sorted(class_stats.items()):
        if stats["total"] == 0 and stats["fp"] == 0:
            continue
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        print(f"{cls_name:<15} | {stats['total']:<7} | {prec*100:>9.2f}% | {rec*100:>9.2f}% | {f1*100:>9.2f}%")

    macro_prec = (np.mean(precisions) * 100.0) if precisions else 0.0
    macro_rec = (np.mean(recalls) * 100.0) if recalls else 0.0
    macro_f1 = (np.mean(f1s) * 100.0) if f1s else 0.0

    print("-" * 65)
    print(f"Macro Precision       : {macro_prec:.2f}%")
    print(f"Macro Recall          : {macro_rec:.2f}%")
    print(f"Macro F1-Score        : {macro_f1:.2f}%")
    print("==========================================================================\n")

    return {
        "accuracy": accuracy,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "total_crops": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate WasteClassifier on ground-truth crops.")
    parser.add_argument(
        "--dataset-dir",
        default="data/trashnet_extracted/dataset-resized",
        help="Path to labeled dataset directory containing class subfolders.",
    )
    parser.add_argument(
        "--model-path",
        default="cashcrow-classification-model/efficientnet_b0_224_5class_int8/models/waste_classifier_fp32.tflite",
        help="Path to TFLite model.",
    )
    parser.add_argument(
        "--classes-path",
        default="cashcrow-classification-model/efficientnet_b0_224_5class_int8/classes.json",
        help="Path to classes.json.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    crops_and_labels = load_ground_truth_crops(dataset_path, samples_per_class=25)

    if not crops_and_labels:
        logger.error("No valid crop samples loaded for evaluation. Check dataset directory.")
        return

    evaluate_classifier(crops_and_labels, args.model_path, args.classes_path)


if __name__ == "__main__":
    main()
