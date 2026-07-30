# Cashcrow Smartbin — AI-Powered Waste Detection & Classification Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](#license)
[![Build & Tests](https://img.shields.io/badge/tests-68%20passed-brightgreen.svg)](#running-tests)

**Cashcrow Smartbin** is an edge-optimized computer vision pipeline for real-time waste item detection, hand-item association, and high-accuracy classification. Designed for smart bin automation (e.g. Jetson Orin Nano / edge devices), it combines motion-triggered gating, YOLO object detection/location, MediaPipe hand tracking, an EfficientNet-B0 TFLite second-stage refiner, and sliding-window consensus voting.

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
│  - Refines/overrides predicted labels (plastic, paper, metal, organic_waste, none)      │
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
- **Class-Agnostic Locator Mode**: Supports running pretrained YOLO models (e.g., `yolo11n.pt`) purely as generic object locators, letting the EfficientNet TFLite refiner perform high-accuracy waste classification on object crops.
- **Second-Stage EfficientNet Refiner**: 96.59% accuracy TFLite classifier fine-tuned on waste categories (`plastic`, `paper`, `metal`, `organic_waste`, `none`).
- **MediaPipe Hand Tracking**: Real-time hand landmark detection and tracking that associates waste objects directly with hands.
- **ByteTrack Multi-Object Tracking**: Fast tracking without ReID overhead, optimized for single/multi-item bin insertions.
- **Consensus-Conditioned Voting**: Aggregates predictions across frames, filtering out transient noise or partial occlusions before finalizing decisions.
- **Real-Time Actuation Hooks**: JSONL decision logging and HTTP webhook hooks for controlling bin motors/servos.

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
├── tests/                          # Offline Pytest unit & integration suite
├── config.yaml                     # Default configuration parameters
├── main.py                         # CLI entry point
├── train_waste_model.py            # Fine-tuning script (TrashNet / TACO / Cashcrow)
├── benchmark_model.py              # Performance benchmarking script
├── requirements.txt                # Requirements file
└── README.md
```

---

## Installation & Setup

### Prerequisites

- **Python 3.9+** (Tested up to Python 3.13)
- OpenCV, PyTorch / Ultralytics, MediaPipe
- (Optional) CUDA-compatible GPU for accelerated inference

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

If fine-tuned weights (`best.pt`) are present in the repository root:

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
| `--weights` | `best.pt` | Path to YOLO model weights |
| `--confidence` | `0.25` | Detection confidence threshold (0.0 – 1.0) |
| `--class-agnostic` | `False` | Run YOLO as generic object locator & refine with EfficientNet |
| `--allow-generic-model` | `False` | Allow running with generic COCO model weights |
| `--track-hands` | `False` | Enable hand tracking and object association |
| `--hand-roi` | `False` | Enable ROI cropping around hands |
| `--show` | `False` | Display live annotated OpenCV preview window |
| `--log-level` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--dry-run` | `False` | Validate configuration and model weights, then exit |

---

## Running Tests

The test suite runs completely offline without requiring a GPU or webcam:

```bash
# Run all unit and integration tests
python -m pytest tests/ -v

# Run detector filter tests
python -m pytest tests/test_detector_filter.py -v
```

---

## Configuration (`config.yaml`)

Key parameters in `config.yaml`:

```yaml
model:
  weights: "best.pt"
  confidence_threshold: 0.25
  class_agnostic: false
  allowed_classes:
    - plastic
    - paper
    - metal
    - glass
    - e-waste
    - organic
    - other

refiner:
  enabled: true
  model_path: "cashcrow-classification-model/efficientnet_b0_224_5class_int8/models/waste_classifier_fp32.tflite"
  classes_path: "cashcrow-classification-model/efficientnet_b0_224_5class_int8/classes.json"
  confidence_threshold: 0.25

hand_tracking:
  enabled: true
  backend: "mediapipe"
  confidence_threshold: 0.3
  max_hand_distance_px: 200.0

webhook:
  url: null  # e.g., "http://localhost:8080/api/decision"
  timeout: 5.0
  max_retries: 3
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
