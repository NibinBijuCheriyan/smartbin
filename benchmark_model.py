import os
import time
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to PyTorch weights")
    parser.add_argument("--dataset", type=str, default="data/dataset.yaml", help="Path to dataset.yaml")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights file '{weights_path}' not found. Please run training first.")
        sys.exit(1)

    print("=" * 60)
    print("Benchmarking Cashcrow Waste Detection Model")
    print("=" * 60)

    # 1. Load PyTorch model
    print("Loading PyTorch model...")
    pt_model = YOLO(weights_path)

    # 2. Export to ONNX
    print("Exporting model to ONNX...")
    onnx_path_str = pt_model.export(format="onnx", imgsz=320, opset=12)
    onnx_path = Path(onnx_path_str)
    print(f"ONNX model exported to {onnx_path}")

    # 3. Model Size Comparison
    pt_size = weights_path.stat().st_size / (1024 * 1024)
    onnx_size = onnx_path.stat().st_size / (1024 * 1024)
    print(f"PyTorch Model Size: {pt_size:.2f} MB")
    print(f"ONNX Model Size:    {onnx_size:.2f} MB")

    # 4. Run PyTorch validation to obtain per-class metrics
    print("\nRunning validation evaluation on validation split...")
    val_results = pt_model.val(data=args.dataset, imgsz=320, split="val", verbose=False)
    
    # Extract metrics
    class_names = val_results.names
    precision_per_class = val_results.box.p
    recall_per_class = val_results.box.r
    maps_per_class = val_results.box.maps  # mAP50-95

    # 5. Speed Benchmarking (PyTorch CPU vs ONNX CPU)
    # Get test image from validation set
    val_images_dir = Path("data/dataset/images/val")
    val_images = list(val_images_dir.glob("*.jpg"))
    if not val_images:
        print("Warning: No validation images found. Generating a blank image for benchmarking.")
        import numpy as np
        import cv2
        dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.imwrite("dummy_bench.jpg", dummy_img)
        bench_img_path = "dummy_bench.jpg"
    else:
        bench_img_path = str(val_images[0])

    print("\nRunning inference speed benchmark on CPU...")
    
    # PyTorch Speed
    print("Warm-up PyTorch model...")
    for _ in range(50):
        _ = pt_model.predict(bench_img_path, imgsz=320, verbose=False)
    
    print("Measuring PyTorch inference speed...")
    t0 = time.perf_counter()
    num_runs = 100
    for _ in range(num_runs):
        _ = pt_model.predict(bench_img_path, imgsz=320, verbose=False)
    t1 = time.perf_counter()
    pt_fps = num_runs / (t1 - t0)
    pt_ms = ((t1 - t0) / num_runs) * 1000

    # ONNX Speed
    print("Loading ONNX model...")
    onnx_model = YOLO(onnx_path)
    print("Warm-up ONNX model...")
    for _ in range(50):
        _ = onnx_model.predict(bench_img_path, imgsz=320, verbose=False)
    
    print("Measuring ONNX inference speed...")
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = onnx_model.predict(bench_img_path, imgsz=320, verbose=False)
    t1 = time.perf_counter()
    onnx_fps = num_runs / (t1 - t0)
    onnx_ms = ((t1 - t0) / num_runs) * 1000

    # Clean dummy image if created
    if Path("dummy_bench.jpg").exists():
        os.remove("dummy_bench.jpg")

    # 6. Generate Report
    report_lines = []
    report_lines.append("# Cashcrow Waste Detection Model Benchmark Report\n")
    report_lines.append("## Model Size")
    report_lines.append(f"- **PyTorch (`best.pt`)**: {pt_size:.2f} MB")
    report_lines.append(f"- **ONNX (`best.onnx`)**: {onnx_size:.2f} MB\n")
    
    report_lines.append("## Inference Performance (CPU)")
    report_lines.append(f"- **PyTorch CPU**: {pt_fps:.2f} FPS ({pt_ms:.2f} ms/frame)")
    report_lines.append(f"- **ONNX CPU**: {onnx_fps:.2f} FPS ({onnx_ms:.2f} ms/frame)\n")

    report_lines.append("## Per-Class Accuracy Metrics")
    report_lines.append("| Class ID | Class Name | Precision | Recall | mAP50-95 |")
    report_lines.append("|---|---|---|---|---|")
    
    underperforming = []
    for idx, name in class_names.items():
        p = precision_per_class[idx] if idx < len(precision_per_class) else 0.0
        r = recall_per_class[idx] if idx < len(recall_per_class) else 0.0
        map_val = maps_per_class[idx] if idx < len(maps_per_class) else 0.0
        report_lines.append(f"| {idx} | {name} | {p:.4f} | {r:.4f} | {map_val:.4f} |")
        
        # Consider underperforming if Precision or Recall is below 0.5
        if p < 0.5 or r < 0.5:
            underperforming.append((name, p, r))

    report_lines.append("\n## Under-performing Classes & Data Gaps")
    if underperforming:
        report_lines.append("The following classes are underperforming (accuracy metrics < 50%) and require more representative, high-quality Cashcrow bin images:")
        for name, p, r in underperforming:
            report_lines.append(f"- **{name}**: Precision={p:.4f}, Recall={r:.4f}")
    else:
        report_lines.append("All classes perform above 50% on validation data. However, for real-world deployment, classes like `e-waste` and `organic` should receive additional target-domain bin images due to dataset sparseness in TACO.")

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
