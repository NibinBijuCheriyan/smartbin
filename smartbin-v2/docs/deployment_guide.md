# SmartBin AI v2 — Edge Hardware Deployment Guide

## 1. Target Hardware Specifications

### Primary Tier: Raspberry Pi 5 (8GB)
- **SoC**: Broadcom BCM2712 quad-core Cortex-A76 @ 2.4GHz
- **RAM**: 8GB LPDDR4X-4267
- **Camera**: Raspberry Pi Camera Module 3 (Sony IMX708 with Autofocus)
- **Target Inference Engine**: ONNX Runtime (CPU provider with ARM NEON) or OpenVINO ARM64
- **Performance**: 20–25 FPS (~40–48 ms total latency)

### Scaling Tier: NVIDIA Jetson Orin Nano / Nano
- **Target Inference Engine**: TensorRT (FP16 engine)
- **Performance**: 45–60 FPS (< 20 ms total latency)

---

## 2. Production Installation Steps

### Step 1: Clone Repository and Create Virtual Environment
```bash
git clone https://github.com/NibinBijuCheriyan/smartbin.git
cd smartbin
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r smartbin-v2/requirements.txt
pip install -e .
```

### Step 2: Model Weights Installation
Place the exported model weights in `smartbin-v2/models/`:
- `smartbin-v2/models/smartbin_yolo11s.onnx`
- `smartbin-v2/models/material_classifier.tflite`
- `smartbin-v2/models/contamination_classifier.tflite`

### Step 3: Hardware Diagnostics Check
Verify camera and serial links before launching daemon:
```bash
python -m smartbin_v2.raspberry_pi.hardware_test
```

### Step 4: Install Systemd Autostart Service
```bash
sudo ./smartbin-v2/deployment/systemd/install_service.sh
sudo systemctl start smartbin
sudo systemctl status smartbin
```

---

## 3. Operational Monitoring

- **Live Logs**: `journalctl -u smartbin -f`
- **Dashboard**: `streamlit run smartbin-v2/dashboard/app.py --server.port 8501`
- **REST API**: `uvicorn smartbin_v2.api.main:app --host 0.0.0.0 --port 8000`
