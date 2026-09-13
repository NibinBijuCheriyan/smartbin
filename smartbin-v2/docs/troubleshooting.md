# SmartBin AI v2 — Field Troubleshooting & Diagnostic Guide

## 1. Camera Acquisition Issues

### Symptom: `Failed to open video source 0`
- **Cause**: Camera ribbon disconnected or USB video device node changed (`/dev/video1` instead of `/dev/video0`).
- **Fix**:
  1. Check connected video nodes: `v4l2-ctl --list-devices` or `ls -l /dev/video*`.
  2. For CSI Camera Module 3, verify libcamera: `rpicam-hello -t 1000`.
  3. Ensure the `pi` user belongs to the `video` group: `sudo usermod -aG video $USER`.

---

## 2. Serial Communication & Servo Faults

### Symptom: `Permission denied: '/dev/ttyACM0'`
- **Cause**: Dialout serial permissions missing for current user.
- **Fix**:
  ```bash
  sudo usermod -aG dialout $USER
  sudo chmod 666 /dev/ttyACM0
  ```

### Symptom: Raspberry Pi crashes or resets during servo actuation
- **Cause**: Servo power drawn from 5V GPIO pin causing voltage brownout.
- **Fix**: Disconnect servo red/black wires from Raspberry Pi. Connect servo directly to an external regulated 5V power adapter (2A minimum) with common ground connected to Arduino GND pin.

---

## 3. Inference & Model Performance

### Symptom: FPS drops below 10 FPS on Raspberry Pi 5
- **Cause**: Running unoptimized PyTorch weights (`best.pt`) on CPU without ONNX / OpenVINO acceleration.
- **Fix**:
  Export to ONNX and update `smartbin-v2/configs/deployment_config.yaml`:
  ```yaml
  runtime:
    inference_backend: "onnx"
    model_path: "smartbin-v2/models/smartbin_yolo11s.onnx"
    num_threads: 4
  ```

### Symptom: False alarms triggered by human hands in bin hopper
- **Cause**: Hard negative background samples not yet ingested into model training.
- **Fix**: Run `python -m smartbin_v2.evaluation.hard_negative_miner` to add hand and empty chute images as background labels, or ensure `hand_tracking.enabled: true` in config.
