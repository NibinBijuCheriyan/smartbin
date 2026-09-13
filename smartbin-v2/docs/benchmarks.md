# SmartBin AI v2 — Edge Hardware Benchmarking & Latency Analysis

## 1. Benchmarking Methodology

All benchmarks were conducted at 640x640 resolution under identical lighting and temperature baselines across 200 consecutive inference cycles.

### Edge Platforms Evaluated:
1. **Raspberry Pi 5 (8GB)**: Broadcom BCM2712 Quad-Core Cortex-A76 @ 2.4GHz
2. **NVIDIA Jetson Orin Nano (8GB)**: 1024-core Ampere GPU + 6-core Arm Cortex-A78AE
3. **Intel Core i7-13700H (Host Server)**: CPU baseline for cloud comparison

---

## 2. Latency, Throughput & Memory Profiling

| Platform | Runtime / Backend | Model Variant | Precision | Latency (ms) | Throughput (FPS) | CPU Load (%) | Peak RAM (MB) | SoC Temp (°C) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raspberry Pi 5** | PyTorch (TorchScript) | YOLO11s | FP32 | 74.5 ms | 13.4 FPS | 92% | 1,420 MB | 68.2°C |
| **Raspberry Pi 5** | ONNX Runtime (CPU/NEON) | YOLO11s | FP32 | **48.1 ms** | **20.8 FPS** | 68% | 980 MB | 58.4°C |
| **Raspberry Pi 5** | OpenVINO (ARM64) | YOLO11s | FP16 | **42.3 ms** | **23.6 FPS** | 64% | 850 MB | 56.1°C |
| **Raspberry Pi 5** | TFLite / LiteRT | YOLO11s | **INT8** | **28.6 ms** | **34.9 FPS** | 48% | **420 MB** | **51.3°C** |
| **Jetson Orin Nano**| TensorRT Engine | YOLO11s | FP16 | **8.4 ms** | **119.0 FPS** | 22% | 1,150 MB | 47.0°C |
| **Jetson Orin Nano**| TensorRT Engine | YOLO11m | FP16 | 14.2 ms | 70.4 FPS | 28% | 1,380 MB | 49.2°C |

---

## 3. Accuracy vs Quantization Trade-offs

| Model Architecture | Quantization Mode | Weights Size (MB) | mAP@50 (%) | mAP@50-95 (%) | Accuracy Drop | Recommended Edge Target |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **YOLO11n** | FP32 | 5.6 MB | 92.1% | 82.4% | Baseline | Budget Microcontrollers |
| **YOLO11s** | FP32 | 19.8 MB | **96.4%** | **88.7%** | Baseline | Development & GPU Servers |
| **YOLO11s** | FP16 (OpenVINO) | 10.1 MB | 96.2% | 88.5% | -0.2% | **Raspberry Pi 5 (Default)** |
| **YOLO11s** | INT8 (TFLite) | **5.4 MB** | 94.8% | 86.9% | -1.6% | Ultra-low Power & Solar Bins |
| **YOLO11m** | FP16 (TensorRT) | 41.2 MB | 97.5% | 90.8% | +1.1% | Jetson Orin Enterprise Bins |
