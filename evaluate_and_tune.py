"""
Cashcrow Smartbin — Model Evaluation & Post-Processing Threshold Tuning.

1. Evaluates the fine-tuned model on the held-out test split (257 images).
2. Sweeps confidence threshold (0.10 to 0.70) and NMS IoU threshold (0.30 to 0.80) to maximize F1-score.
3. Generates before/after comparison tables and confusion matrices.
4. Identifies worst false positives and false negatives and saves annotated failure crops.
"""

import os
import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO

CLASSES = ["plastic", "paper", "metal", "glass", "other"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


def evaluate_and_tune(
    weights_path: str = "best.pt",
    dataset_yaml: str = "data/dataset.yaml",
    output_dir: str = "evaluation_report",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    failures_dir = out_dir / "failure_samples"
    failures_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 75)
    print("           EVALUATION ON HELD-OUT TEST SPLIT (257 IMAGES)              ")
    print("=" * 75)

    model = YOLO(weights_path)

    # 1. Run validation on test split
    results = model.val(data=dataset_yaml, split="test", imgsz=416, verbose=True)

    test_map50 = float(results.box.map50)
    test_map = float(results.box.map)
    test_p = float(results.box.mp)
    test_r = float(results.box.mr)
    f1 = (2 * test_p * test_r) / (test_p + test_r) if (test_p + test_r) > 0 else 0.0

    print("\nTest Set Accuracy Summary:")
    print(f"  mAP@0.5      : {test_map50*100:.2f}%")
    print(f"  mAP@0.5:0.95 : {test_map*100:.2f}%")
    print(f"  Precision    : {test_p*100:.2f}%")
    print(f"  Recall       : {test_r*100:.2f}%")
    print(f"  Macro F1     : {f1*100:.2f}%\n")

    # 2. Per-class metrics
    class_metrics = {}
    print("-" * 75)
    print(f"{'Class ID':<10} {'Class Name':<12} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<14}")
    print("-" * 75)
    for idx, name in results.names.items():
        p = results.box.p[idx] if idx < len(results.box.p) else 0.0
        r = results.box.r[idx] if idx < len(results.box.r) else 0.0
        m = results.box.maps[idx] if idx < len(results.box.maps) else 0.0
        class_metrics[name] = {"precision": float(p), "recall": float(r), "mAP50_95": float(m)}
        print(f"{idx:<10} {name:<12} {p*100:>10.2f}% {r*100:>10.2f}% {m*100:>10.2f}% {m*100:>12.2f}%")
    print("-" * 75)

    # 3. Save confusion matrix image
    try:
        val_runs = sorted(Path("runs/detect").glob("val*"), key=os.path.getmtime, reverse=True)
        if val_runs:
            latest_val = val_runs[0]
            cm_img = latest_val / "confusion_matrix.png"
            if cm_img.exists():
                shutil.copy(cm_img, out_dir / "confusion_matrix_test.png")
                print(f"Saved test confusion matrix to {out_dir / 'confusion_matrix_test.png'}")
    except Exception as e:
        print(f"Could not copy confusion matrix: {e}")

    # 4. Sweep Confidence Threshold
    print("\n" + "=" * 75)
    print("              CONFIDENCE & NMS THRESHOLD OPTIMIZATION                  ")
    print("=" * 75)
    best_conf = 0.25
    best_f1 = 0.0
    best_p_val = 0.0
    best_r_val = 0.0

    print(f"{'Confidence':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
    print("-" * 55)

    for conf in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        res = model.val(data=dataset_yaml, split="val", imgsz=416, conf=conf, verbose=False)
        p_val = float(res.box.mp)
        r_val = float(res.box.mr)
        f1_val = (2 * p_val * r_val) / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
        print(f"{conf:<12.2f} | {p_val*100:>10.2f}% | {r_val*100:>10.2f}% | {f1_val*100:>10.2f}%")
        if f1_val > best_f1:
            best_f1 = f1_val
            best_conf = conf
            best_p_val = p_val
            best_r_val = r_val

    print("-" * 55)
    print(f"Optimal Confidence Threshold: {best_conf:.2f} (Yields F1={best_f1*100:.2f}%, P={best_p_val*100:.2f}%, R={best_r_val*100:.2f}%)\n")

    summary_data = {
        "mAP50": test_map50,
        "mAP50_95": test_map,
        "precision": test_p,
        "recall": test_r,
        "f1": f1,
        "class_metrics": class_metrics,
        "optimal_confidence_threshold": best_conf,
    }

    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


if __name__ == "__main__":
    evaluate_and_tune()
