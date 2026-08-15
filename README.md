# Cashcrow Smartbin — AI-Powered Waste Detection & Classification Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-orange.svg)](https://pytorch.org/)
[![Build & Tests](https://img.shields.io/badge/tests-77%20passed-brightgreen.svg)](#running-tests)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](#license)

**Cashcrow Smartbin** is an enterprise-grade computer vision pipeline designed for real-time waste item detection, hand-item association, and high-accuracy multi-stage classification on edge devices (e.g., NVIDIA Jetson Orin Nano, edge IPCs). 

It combines motion-triggered gating, YOLO v11 object detection/location, MediaPipe ML hand landmark tracking, an EfficientNet-B0 TFLite second-stage refiner, and sliding-window consensus voting.

---

## Architecture & Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INPUT FRAME STREAM                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: MOTION TRIGGER (IDLE ↔ ACTIVE)                         │
│  - Motion gating via frame differencing                                                │
│  - Dormant mode saves ~95% compute when bin is idle                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                    (Triggered / Hand)
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           STAGE 2: HAND DETECTION & TRACKING                            │
│  - MediaPipe ML Hand Tracker (robust across skin tones & lighting)                     │
│  - Hand ROI extraction & spatial hand-object association                               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 3: YOLO DETECTOR & BYTETRACK TRACKING                       │
│  - Fine-tuned waste model (best.pt) OR Class-Agnostic Object Locator (--class-agnostic) │
│  - Bounding box extraction & ByteTrack spatial multi-object tracking                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                       (Object Crops)
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                   STAGE 4: EFFICIENTNET-B0 SECOND-STAGE REFINER                         │
│  - TFLite FP32 Classifier (96.59% accuracy) on cropped bounding box images             │
│  - Refines/overrides predicted labels (plastic, paper, metal, glass, other)              │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 5: MAJORITY VOTER & DECISION HOOKS                           │
│  - Sliding-window vote aggregation & consensus confidence rating                        │
│  - Emits JSONL decision logs and optional HTTP Webhook for bin hardware actuation       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Motion-Triggered Gating**: Detector remains dormant until motion or hands are detected, saving ~95% edge compute.
- **CUDA GPU Fine-Tuned Model**: Fine-tuned YOLO model (`best.pt`) optimized for waste categories (`plastic`, `paper`, `metal`, `glass`, `other`).
- **20x Fast GrabCut Data Pipeline**: Automated foreground segmentation pipeline with scale-invariant bounding box generation for custom dataset training.
- **Class-Agnostic Locator Mode**: Supports running pretrained YOLO models (e.g., `yolo11n.pt`) purely as generic object locators, letting the EfficientNet TFLite refiner perform high-accuracy classification on object crops.
- **Second-Stage EfficientNet Refiner**: 96.59% accuracy TFLite classifier supported via `ai-edge-litert`, `tflite-runtime`, or `tensorflow`.
- **MediaPipe Hand Tracking**: Real-time hand landmark detection and spatial tracking that associates waste objects directly with hands.
- **ByteTrack Multi-Object Tracking**: Fast tracking without ReID overhead, optimized for single/multi-item bin insertions.
- **Consensus-Conditioned Voting**: Aggregates predictions across frames, filtering out transient noise or partial occlusions before finalizing decisions.
- **Real-Time Actuation Hooks**: JSONL decision logging and HTTP webhook hooks for controlling bin motors/servos.

---

## Model Benchmark & Validation Metrics

The fine-tuned YOLO waste detection model (`best.pt`) was trained for 50 epochs on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU**:

| Metric | Benchmark Result |
|---|---|
| **Overall mAP@50** | **`0.632`** |
| **Overall mAP@50-95** | **`0.448`** |
| **Precision (P)** | **`0.661`** |
| **Recall (R)** | **`0.581`** |
| **Inference Speed (GPU)** | **`4.2 ms / frame`** (~238 FPS) |

### Per-Class Evaluation Metrics (mAP@50)
- 🍷 **Glass**: `0.800` (P: 0.813, R: 0.684)
- 📄 **Paper**: `0.797` (P: 0.731, R: 0.741)
- 🥫 **Metal**: `0.778` (P: 0.586, R: 0.802)
- 🥤 **Plastic**: `0.632` (P: 0.653, R: 0.563)
- 📦 **Other**: `0.154` (P: 0.524, R: 0.115)

---

## Directory Structure

```
smartbin/
├── smartbin/                       # Core Python Package
│   ├── config.py                   # Config dataclasses, YAML loader & CLI overrides
│   ├── trigger.py                  # Motion gate (Frame diffing)
│   ├── state_machine.py            # IDLE ↔ ACTIVE state machine & buffer
│   ├── detector.py                 # YOLO + ByteTrack wrapper & Class-Agnostic mode
│   ├── hand_tracker.py             # MediaPipe / Skin-color hand tracker & association
│   ├── refiner.py                  # EfficientNet-B0 TFLite second-stage classifier
│   ├── voter.py                    # Sliding-window majority voting & consensus
│   ├── decision.py                 # Decision events & output hooks (JSONL, Webhook)
│   └── pipeline.py                 # Pipeline orchestrator
├── cashcrow-classification-model/  # EfficientNet-B0 TFLite model & vocabulary
├── tests/                          # Offline Pytest unit & integration suite (77 tests)
├── config.yaml                     # Default configuration parameters
├── main.py                         # CLI entry point
├── train_waste_model.py            # GPU fine-tuning script (TrashNet / TACO / Cashcrow)
├── audit_dataset.py                # Dataset audit and integrity verification script
├── benchmark_model.py              # Performance benchmarking script
├── requirements.txt                # Dependencies (ultralytics, opencv, ai-edge-litert, etc.)
└── README.md
```

---

## Installation & Setup

### Prerequisites

- **Python 3.9+** (Tested up to Python 3.13)
- OpenCV, PyTorch / Ultralytics, MediaPipe, `ai-edge-litert`
- (Recommended) NVIDIA CUDA-compatible GPU for accelerated inference and training

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/NibinBijuCheriyan/smartbin.git
cd smartbin

# Install required dependencies
pip install -r requirements.txt
```

---

## Quick Start & Usage

### 1. Run with Fine-Tuned Model (`best.pt`)

```bash
python main.py --weights best.pt --show
```

### 2. Run in Class-Agnostic Mode (Generic YOLO + EfficientNet Refiner)

If using pretrained COCO weights (e.g., `yolo11n.pt`), enable `--class-agnostic` so YOLO locates candidate objects while the EfficientNet classifier assigns waste labels:

```bash
python main.py --weights yolo11n.pt --allow-generic-model --class-agnostic --confidence 0.15 --show
```

### 3. Run on Video File

```bash
python main.py --source path/to/video.mp4 --show
```

### 4. Dry Run (Validate Config & Model Loading)

```bash
python main.py --dry-run
```

---

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--config` | `config.yaml` | Path to YAML config file |
| `--source` | `0` (webcam) | Video source (camera index `0` or file path) |
| `--weights` | `best.pt` | Path to YOLO model weights (auto-falls back to `yolo11n.pt` + class-agnostic mode if missing) |
| `--confidence` | `0.25` | Detection confidence threshold (0.0 – 1.0) |
| `--class-agnostic` | `False` | Run YOLO as generic object locator (bypasses class filtering, lets EfficientNet refiner classify crops) |
| `--allow-generic-model` | `False` | Allow running with generic COCO model weights (automatically enables `--class-agnostic`) |
| `--track-hands` | `False` | Enable hand tracking and object association |
| `--hand-roi` | `False` | Enable ROI cropping around hands |
| `--show` | `False` | Display live annotated OpenCV preview window |
| `--log-level` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--dry-run` | `False` | Validate configuration and model weights, then exit |

---

## Training Custom Weights

To train or fine-tune the YOLO waste detection model on your GPU:

```bash
# Train on GPU using CUDA
python train_waste_model.py --device cuda:0 --epochs 50 --patience 10 --allow-partial --allow-sparse

# Audit dataset distribution and label integrity
python audit_dataset.py
```

Features of the training script:
- Automated GrabCut segmentation for studio datasets (TrashNet).
- TACO annotation remapping into 5 Cashcrow target waste categories.
- Automatic device selection (`cuda:0` / `mps` / `cpu`).
- Automatic export of fine-tuned weights to `best.pt`.

---

## Running Tests

The test suite runs completely offline without requiring a GPU or webcam:

```bash
# Run all unit and integration tests (77 passed)
python -m pytest tests/ -v
```

---

## Decision Outputs & Webhooks

When a waste decision is finalized by the voter, it is automatically written to `decisions.jsonl` and sent to any configured HTTP Webhook:

```json
{
  "track_id": 1,
  "item_class": "plastic",
  "confidence": 0.942,
  "frame_count": 12,
  "total_frames": 15,
  "is_certain": true,
  "timestamp": "2026-07-30T21:10:35Z",
  "hand_id": 1,
  "is_held_by_hand": true
}
```

---

## License

Proprietary — Cashcrow Technologies.
