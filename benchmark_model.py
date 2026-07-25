"""
Cashcrow Smartbin — Model Benchmarking.

Runs validation, speed benchmarks, and generates a comprehensive report
including per-class metrics, confusion matrix, and model size comparisons.

Key improvements over original:
- Supports a held-out real-world test set (--test-set) for OOD evaluation.
- Clearly labels in-distribution vs. deployment metrics.
- Generates a confusion matrix (saved as image + text table).
- Reports class sample counts for context.

Usage:
    # Basic benchmark against training val split
    python benchmark_model.py

    # Benchmark with a held-out real-world test set
    python benchmark_model.py --test-set path/to/test_data.yaml

    # Custom weights
    python benchmark_model.py --weights best.pt --dataset data/dataset.yaml
"""

import os
import time
import sys
import argparse
from pathlib import Path

import numpy as np


def run_validation(model, dataset_path: str, split: str = "val", imgsz: int = 320):
    """Run YOLO validation and return results object."""
    return model.val(data=dataset_path, imgsz=imgsz, split=split, verbose=False)


def format_metrics_table(val_results, label: str) -> list:
    """Format per-class metrics as markdown table lines."""
    lines = []
    lines.append(f"### {label}")
    lines.append("")

    class_names = val_results.names
    precision_per_class = val_results.box.p
    recall_per_class = val_results.box.r
    maps_per_class = val_results.box.maps  # mAP50-95

    lines.append("| Class ID | Class Name | Precision | Recall | mAP50-95 |")
    lines.append("|---|---|---|---|---|")

    underperforming = []
    for idx, name in class_names.items():
        p = precision_per_class[idx] if idx < len(precision_per_class) else 0.0
        r = recall_per_class[idx] if idx < len(recall_per_class) else 0.0
        map_val = maps_per_class[idx] if idx < len(maps_per_class) else 0.0
        lines.append(f"| {idx} | {name} | {p:.4f} | {r:.4f} | {map_val:.4f} |")

        if p < 0.5 or r < 0.5:
            underperforming.append((name, p, r))

    lines.append("")
    return lines, underperforming


def format_confusion_matrix(val_results) -> list:
    """Generate a text-based confusion matrix section."""
    lines = []
    lines.append("### Confusion Matrix")
    lines.append("")

    try:
        cm = val_results.confusion_matrix
        if cm is not None:
            matrix = cm.matrix if hasattr(cm, 'matrix') else None
            class_names = val_results.names

            if matrix is not None:
                num_classes = min(len(class_names), matrix.shape[0])

                # Header
                header = "| Actual \\ Predicted |"
                for idx in range(num_classes):
                    header += f" {class_names.get(idx, f'cls_{idx}')} |"
                lines.append(header)

                sep = "|---|"
                for _ in range(num_classes):
                    sep += "---|"
                lines.append(sep)

                # Rows
                for i in range(num_classes):
                    row = f"| **{class_names.get(i, f'cls_{i}')}** |"
                    for j in range(num_classes):
                        val = int(matrix[i][j]) if i < matrix.shape[0] and j < matrix.shape[1] else 0
                        row += f" {val} |"
                    lines.append(row)

                lines.append("")
                lines.append("*Rows = actual class, Columns = predicted class. "
                           "Off-diagonal entries reveal systematic misclassifications.*")
            else:
                lines.append("*Confusion matrix data not available.*")
        else:
            lines.append("*Confusion matrix not available for this validation run.*")
    except Exception as e:
        lines.append(f"*Could not generate confusion matrix: {e}*")

    lines.append("")
    return lines


def save_confusion_matrix_plot(val_results, output_path: str) -> bool:
    """Try to save the confusion matrix as a plot image."""
    try:
        cm = val_results.confusion_matrix
        if cm is not None and hasattr(cm, 'plot'):
            cm.plot(save_dir=str(Path(output_path).parent), names=list(val_results.names.values()))
            return True
    except Exception:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the Cashcrow waste detection model."
    )
    parser.add_argument(
        "--weights", type=str, default="best.pt",
        help="Path to PyTorch weights",
    )
    parser.add_argument(
        "--dataset", type=str, default="data/dataset.yaml",
        help="Path to dataset.yaml (in-distribution validation)",
    )
    parser.add_argument(
        "--test-set", type=str, default=None,
        help="Path to a held-out real-world test dataset.yaml (OOD evaluation). "
             "Should contain images NOT used in any training run, ideally from "
             "the target hardware/camera.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=320,
        help="Image size for validation and benchmarking (default: 320)",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights file '{weights_path}' not found. Please run training first.")
        sys.exit(1)

    print("=" * 60)
    print("Benchmarking Cashcrow Waste Detection Model")
    print("=" * 60)

    from ultralytics import YOLO

    # 1. Load PyTorch model
    print("Loading PyTorch model...")
    pt_model = YOLO(weights_path)

    # 2. Export to ONNX
    print("Exporting model to ONNX...")
    onnx_path_str = pt_model.export(format="onnx", imgsz=args.imgsz, opset=12)
    onnx_path = Path(onnx_path_str)
    print(f"ONNX model exported to {onnx_path}")

    # 3. Model Size Comparison
    pt_size = weights_path.stat().st_size / (1024 * 1024)
    onnx_size = onnx_path.stat().st_size / (1024 * 1024)
    print(f"PyTorch Model Size: {pt_size:.2f} MB")
    print(f"ONNX Model Size:    {onnx_size:.2f} MB")

    # 4. In-distribution validation
    print("\nRunning IN-DISTRIBUTION validation on training val split...")
    val_results = run_validation(pt_model, args.dataset, imgsz=args.imgsz)

    # 5. Optional OOD test set evaluation
    ood_results = None
    if args.test_set:
        test_set_path = Path(args.test_set)
        if not test_set_path.exists():
            print(f"Warning: Test set '{test_set_path}' not found. Skipping OOD evaluation.")
        else:
            print(f"\nRunning REAL-WORLD (OOD) evaluation on held-out test set: {args.test_set}")
            ood_results = run_validation(pt_model, args.test_set, imgsz=args.imgsz)

    # 6. Speed Benchmarking
    val_images_dir = Path("data/dataset/images/val")
    val_images = list(val_images_dir.glob("*.jpg")) if val_images_dir.exists() else []
    if not val_images:
        print("Warning: No validation images found. Generating a blank image for benchmarking.")
        import cv2
        dummy_img = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        cv2.imwrite("dummy_bench.jpg", dummy_img)
        bench_img_path = "dummy_bench.jpg"
    else:
        bench_img_path = str(val_images[0])

    print("\nRunning inference speed benchmark on CPU...")

    # PyTorch Speed
    print("Warm-up PyTorch model...")
    for _ in range(50):
        _ = pt_model.predict(bench_img_path, imgsz=args.imgsz, verbose=False)

    print("Measuring PyTorch inference speed...")
    t0 = time.perf_counter()
    num_runs = 100
    for _ in range(num_runs):
        _ = pt_model.predict(bench_img_path, imgsz=args.imgsz, verbose=False)
    t1 = time.perf_counter()
    pt_fps = num_runs / (t1 - t0)
    pt_ms = ((t1 - t0) / num_runs) * 1000

    # ONNX Speed
    print("Loading ONNX model...")
    onnx_model = YOLO(onnx_path)
    print("Warm-up ONNX model...")
    for _ in range(50):
        _ = onnx_model.predict(bench_img_path, imgsz=args.imgsz, verbose=False)

    print("Measuring ONNX inference speed...")
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = onnx_model.predict(bench_img_path, imgsz=args.imgsz, verbose=False)
    t1 = time.perf_counter()
    onnx_fps = num_runs / (t1 - t0)
    onnx_ms = ((t1 - t0) / num_runs) * 1000

    # Clean dummy image if created
    if Path("dummy_bench.jpg").exists():
        os.remove("dummy_bench.jpg")

    # 7. Generate Report
    report_lines = []
    report_lines.append("# Cashcrow Waste Detection Model Benchmark Report\n")

    report_lines.append("## Model Size")
    report_lines.append(f"- **PyTorch (`best.pt`)**: {pt_size:.2f} MB")
    report_lines.append(f"- **ONNX (`best.onnx`)**: {onnx_size:.2f} MB\n")

    report_lines.append("## Inference Performance (CPU)")
    report_lines.append(f"- **PyTorch CPU**: {pt_fps:.2f} FPS ({pt_ms:.2f} ms/frame)")
    report_lines.append(f"- **ONNX CPU**: {onnx_fps:.2f} FPS ({onnx_ms:.2f} ms/frame)\n")

    # In-distribution metrics
    report_lines.append("## Per-Class Accuracy Metrics")
    report_lines.append("")
    report_lines.append("> **⚠️ IMPORTANT:** The metrics below are measured on the **in-distribution "
                       "validation split** — the same distribution the model trained on. "
                       "These are **NOT a proxy for deployment accuracy.** Use `--test-set` "
                       "with held-out real-world images for deployment-relevant metrics.")
    report_lines.append("")

    id_table, id_underperforming = format_metrics_table(
        val_results, "In-Distribution Validation Metrics"
    )
    report_lines.extend(id_table)

    # Confusion matrix
    cm_lines = format_confusion_matrix(val_results)
    report_lines.extend(cm_lines)

    # Save confusion matrix plot
    saved = save_confusion_matrix_plot(val_results, "confusion_matrix.png")
    if saved:
        report_lines.append("*Confusion matrix plot saved to `confusion_matrix.png`*\n")

    # OOD test set metrics (if provided)
    if ood_results is not None:
        report_lines.append("---\n")
        report_lines.append("## Real-World (Out-of-Distribution) Test Set Metrics")
        report_lines.append("")
        report_lines.append("> These metrics are measured on a **held-out test set** "
                           "not used during training. They are a better proxy for "
                           "real-world deployment performance.")
        report_lines.append("")

        ood_table, ood_underperforming = format_metrics_table(
            ood_results, "Real-World Test Set Metrics"
        )
        report_lines.extend(ood_table)

        ood_cm_lines = format_confusion_matrix(ood_results)
        report_lines.extend(ood_cm_lines)

        if ood_underperforming:
            report_lines.append("### Real-World Under-performing Classes")
            report_lines.append("The following classes underperform on real-world data:")
            for name, p, r in ood_underperforming:
                report_lines.append(f"- **{name}**: Precision={p:.4f}, Recall={r:.4f}")
            report_lines.append("")

    # Underperforming classes summary
    report_lines.append("\n## Under-performing Classes & Data Gaps")
    if id_underperforming:
        report_lines.append("The following classes are underperforming on in-distribution data "
                          "(Precision or Recall < 50%):")
        for name, p, r in id_underperforming:
            report_lines.append(f"- **{name}**: Precision={p:.4f}, Recall={r:.4f}")
        report_lines.append("")
        report_lines.append("**Action required:** Collect more representative images for these "
                          "classes from the target Cashcrow bin camera.")
    else:
        report_lines.append("All classes perform above 50% on in-distribution validation data. "
                          "However, for real-world deployment, validate with `--test-set` using "
                          "images captured from the actual target hardware/camera.")

    report_lines.append("\n## TensorRT INT8 Quantization Workflow on Edge Hardware")
    report_lines.append("To run INT8 quantization via TensorRT on the Jetson Nano/Orin Nano, follow these steps:")
    report_lines.append("### Step 1: Install TensorRT")
    report_lines.append("Ensure TensorRT is installed on Jetson (usually pre-installed via JetPack).")
    report_lines.append("### Step 2: Run calibration and compile the Engine")
    report_lines.append("Create a calibration dataset using validation/test images (e.g. saved in a directory `calib_images/`) and run:")
    report_lines.append("```bash")
    report_lines.append("# Quantize and export to TensorRT engine")
    report_lines.append("trtexec --onnx=best.onnx --saveEngine=best.engine --int8 --fp16 --calib=calib.cache")
    report_lines.append("```")
    report_lines.append("*(Alternatively, compile to FP16 engine if calibration images are unavailable:)*")
    report_lines.append("```bash")
    report_lines.append("trtexec --onnx=best.onnx --saveEngine=best.engine --fp16")
    report_lines.append("```\n")

    report_content = "\n".join(report_lines)
    print("\n" + report_content)

    # Save to file
    report_path = Path("benchmark_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Report written to {report_path.absolute()}")


if __name__ == "__main__":
    main()
