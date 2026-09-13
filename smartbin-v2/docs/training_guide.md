# SmartBin AI v2 — YOLO11 Training & Export Guide

This guide covers complete training workflows, hyperparameter optimization, and edge export pipelines for the SmartBin computer vision model.

---

## 1. Quick Start Training

Run default training using YOLO11s on the configured dataset:

```bash
python -m smartbin_v2.training.train --config smartbin-v2/configs/training_config.yaml
```

### CLI Overrides
```bash
# Train lightweight nano model for high FPS edge deployment
python -m smartbin_v2.training.train --model yolo11n.pt --epochs 100 --batch 32

# Multi-GPU training (e.g. 2 GPUs)
python -m smartbin_v2.training.train --model yolo11m.pt --device 0,1 --batch 64
```

---

## 2. Checkpoint Resumption

If training was interrupted due to power loss or preemptions:

```bash
python -m smartbin_v2.training.resume --checkpoint smartbin-v2/runs/train/smartbin_yolo11s/weights/last.pt
```

---

## 3. Automated Hyperparameter Search

Optimize learning rates, weight decay, momentum, and augmentation probabilities using genetic search / Optuna:

```bash
python -m smartbin_v2.training.hyperparameter_search --data smartbin-v2/datasets/smartbin_dataset.yaml --iterations 20 --epochs 15
```

Optimal hyperparameters are saved to `smartbin-v2/experiments/best_hyperparameters.yaml`.

---

## 4. K-Fold Cross Validation

Verify model generalization across 5 stratified folds:

```bash
python -m smartbin_v2.training.kfold_training --folds 5 --epochs 25
```

---

## 5. Model Export for Edge Hardware

Export trained PyTorch checkpoints to optimized edge execution runtimes:

```bash
# Export to ONNX, OpenVINO, and TFLite INT8
python -m smartbin_v2.training.export_model \
    --weights smartbin-v2/runs/train/smartbin_yolo11s/weights/best.pt \
    --formats onnx openvino tflite \
    --int8 \
    --imgsz 640
```

Exported models will be stored in `smartbin-v2/models/`.
