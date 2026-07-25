# Smartbin — AI-Powered Waste Detection Pipeline

**Early-stage prototype** — Not production-ready. Active experimentation with YOLO + ByteTrack for real-time waste classification on edge devices.

Smartbin is a computer vision pipeline designed to detect and classify waste items in real-time using a motion-triggered, multi-frame consensus voting system. Built for low-power edge deployment (Jetson Orin Nano), it combines motion detection, object tracking, and intelligent frame aggregation to make stable waste classification decisions despite occlusion and noise.

---

## What It Does

```
┌─────────────┐     ┌──────────────┐     ┌───────────┐     ┌──────────────┐     ┌─────────────────┐
│  Camera /   │────▶│ Motion Gate  │────▶│   State   │────▶│  YOLO +      │────▶│  Majority Vote  │
│   Video     │     │ (Trigger)    │     │  Machine  │     │  ByteTrack   │     │  + Confidence   │
└─────────────┘     └──────────────┘     └───────────┘     └──────────────┘     └─────────────────┘
                                               │                                          │
                                               │ IDLE ↔ ACTIVE                          │
                                               │                                        ▼
                                               │                              ┌──────────────────┐
                                               └─────────────────────────────▶│  Decision Hooks  │
                                                                              │  (JSONL/Webhook) │
                                                                              └──────────────────┘
```

### Key Design Decisions

- **Motion-triggered gating** — Detector only activates when motion is detected. Saves ~95% compute on idle bins.
- **ByteTrack (not DeepSORT)** — No ReID feature extraction needed. Lightweight, handles hand occlusion well.
- **Consensus-conditioned confidence** — Final confidence is the mean of only the frames that agreed with the winning label. Noisy/occluded frames don't dilute the signal.
- **Clean abstraction boundaries** — Swap YOLO for TensorRT without touching trigger/voter/hook logic.

---

## Current Status

###  What Works
- Motion detection via frame differencing (configurable threshold)
- YOLO11 inference with ByteTrack tracking
- Sliding-window majority voting with per-track histories
- YAML-based configuration with CLI overrides
- JSONL structured output logging
- **HTTP webhook hook** for real actuation/consumption (servo controllers, dashboards)
- **MediaPipe hand tracking** (reliable across skin tones, lighting, and gloved hands)
- **Model class validation** — refuses to run with COCO weights unless explicitly overridden
- **Dry-run mode** — validate config and model without starting the camera loop
- Tests runnable without GPU or camera

###  Known Limitations
- **Dataset domain gap** — TrashNet (studio shots) and TACO (outdoor litter) do not resemble the deployment scene (hand holding item over a bin, indoor lighting, close range). **First-party Cashcrow bin images are required for deployment-quality accuracy.** See [Training](#training-a-waste-model).
- **No TensorRT support yet** — Only Ultralytics YOLO via PyTorch. TensorRT stub present for future Jetson optimization.
- **Limited edge testing** — Developed locally; not yet validated on actual Jetson hardware.

> **⚠️ IMPORTANT:** TrashNet and TACO alone are **not sufficient for deployment**. They serve as pretraining data only. You MUST collect and label images from your actual target camera/hardware for the model to work reliably in production.

---

## Project Structure

```
smartbin/
├── smartbin/                       # Main Python package
│   ├── config.py                   # YAML config loading + CLI merge
│   ├── trigger.py                  # Motion gate (frame differencing)
│   ├── state_machine.py            # IDLE ↔ ACTIVE lifecycle + buffering
│   ├── detector.py                 # YOLO + ByteTrack wrapper (abstracted)
│   ├── hand_tracker.py             # Hand detection (MediaPipe / skin-color)
│   ├── voter.py                    # Sliding-window majority vote
│   ├── decision.py                 # Decision event + hooks (JSONL, Webhook)
│   └── pipeline.py                 # Orchestrator (thin wiring layer)
├── tests/                          # Test suite (runs offline)
├── config.yaml                     # Default configuration (all tunable)
├── main.py                         # CLI entry point
├── train_waste_model.py            # Training pipeline (TrashNet + TACO + Cashcrow)
├── benchmark_model.py              # Benchmarking + metrics report
├── requirements.txt                # Dependencies
└── README.md
```

### Runtime Flow

1. **Main loop** (`pipeline.py`) reads frames from webcam or video file at configurable FPS.
2. **IDLE state** — Motion trigger checks each frame. Dormant mode keeps compute cost near zero.
3. **ACTIVE state (on trigger)** — YOLO detector runs on each frame. ByteTrack assigns/maintains object IDs.
4. **State machine** buffers detections until window fills or no motion for N frames.
5. **Voter** aggregates per-track class predictions via majority vote + consensus confidence.
6. **Decision hooks** output to logging, JSONL, and optional HTTP webhook, then return to IDLE.

---

## Setup

### Prerequisites
- **Python 3.9+**
- (Optional) NVIDIA GPU with CUDA for faster inference
- (Optional) MediaPipe for ML-based hand tracking

### Installation

```bash
# Clone the repository
git clone https://github.com/NibinBijuCheriyan/smartbin.git
cd smartbin

# Install dependencies
pip install -r requirements.txt

# Or install as editable package (for development)
pip install -e ".[dev]"
```

### Model Weights

The pipeline requires **fine-tuned waste model weights** to classify waste correctly. COCO-pretrained weights (e.g., `yolo11n.pt`) will be rejected by default.

```bash
# Train a waste model (see Training section below)
python train_waste_model.py

# Or use --allow-generic-model to run with COCO weights (for testing only)
python main.py --allow-generic-model
```

---

## Usage

### Run against Webcam

```bash
python main.py --show
```

### Run against Video File

```bash
python main.py --source path/to/video.mp4 --show
```

### Custom Model & Confidence

```bash
python main.py --weights best.pt --confidence 0.5 --show
```

### Validate Config (Dry Run)

```bash
python main.py --dry-run
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.yaml` | Path to YAML config file |
| `--source` | `0` (webcam) | Video source: integer for webcam ID, or file path |
| `--weights` | `best.pt` | Path to YOLO model weights |
| `--confidence` | `0.25` | Detection confidence threshold (0–1) |
| `--show` | off | Display annotated live preview window |
| `--log-level` | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |
| `--dry-run` | off | Validate config and model, then exit |
| `--allow-generic-model` | off | Allow running with COCO weights (not recommended) |
| `--track-hands` | off | Enable hand tracking |
| `--hand-roi` | off | Enable hand ROI cropping |

CLI arguments **override** config file values.

---

## Training a Waste Model

The training pipeline downloads TrashNet and TACO datasets and fine-tunes a YOLO model.

```bash
# Full training (downloads TrashNet + TACO, 50 epochs with early stopping)
python train_waste_model.py

# Quick test with synthetic data
python train_waste_model.py --mock --epochs 2

# Include first-party Cashcrow bin images (STRONGLY RECOMMENDED)
python train_waste_model.py --cashcrow-data path/to/labeled/images

# Allow training if only one data source succeeds
python train_waste_model.py --allow-partial
```

### Training CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mock` | off | Use synthetic data (testing only) |
| `--epochs` | `50` | Training epochs |
| `--patience` | `10` | Early stopping patience |
| `--imgsz` | `640` | Training image size |
| `--device` | `auto` | Device: auto, cpu, cuda:0, mps |
| `--cashcrow-data` | none | Path to first-party labeled images |
| `--allow-partial` | off | Continue if one data source fails |
| `--allow-sparse` | off | Allow under-represented classes |
| `--trashnet-mode` | `grabcut` | TrashNet handling: grabcut, drop |

### Dataset Domain Gap

> **TrashNet** images are studio classification shots (one item on a plain background).
> **TACO** images are outdoor litter in varied environments.
>
> Neither resembles the actual deployment scene: a hand holding an item over a bin, indoor lighting, close range. **First-party Cashcrow bin images are essential for deployment accuracy.**

The training pipeline uses **GrabCut foreground segmentation** for TrashNet images instead of naive full-frame bounding boxes, which would train unrealistic box geometry.

---

## Benchmarking

```bash
# Basic benchmark
python benchmark_model.py

# With held-out real-world test set (RECOMMENDED)
python benchmark_model.py --test-set path/to/test_data.yaml
```

The benchmark report clearly separates:
- **In-distribution metrics** (training val split) — NOT a proxy for deployment accuracy
- **Real-world metrics** (held-out test set) — actual deployment performance estimate
- **Confusion matrix** — reveals systematic misclassifications (e.g., glass vs. plastic)

---

## Hand Tracking

Two backends are available:

### MediaPipe (Recommended)
ML-based hand detection. Reliable across skin tones, lighting conditions, and gloved hands.

```yaml
hand_tracking:
  enabled: true
  backend: "mediapipe"
```

### Skin Color (Legacy Fallback)
HSV/YCrCb skin-tone thresholding. Only for extremely constrained hardware.

**Known failure modes:**
- Unreliable across skin tones (tuned for a narrow band)
- False positives on skin-colored objects (wood, cardboard, tan plastic)
- Fails under non-standard lighting
- Cannot detect gloved hands

```yaml
hand_tracking:
  enabled: true
  backend: "skin_color"
```

---

## Decision Hooks

### Built-in Hooks

| Hook | Description |
|------|-------------|
| `LoggingHook` | Logs decisions at INFO level (always enabled) |
| `JsonlFileHook` | Appends JSON lines to a file (configurable path) |
| `WebhookHook` | POSTs decisions to an HTTP endpoint (configurable URL) |

### Webhook Hook

The webhook hook closes the loop from "classified item" to "physical or system action."

```yaml
webhook:
  url: "http://localhost:8080/api/decision"
  timeout: 5.0
  max_retries: 3
```

**Contract:**
- Each decision is POSTed individually as JSON.
- Retries on transient failures with exponential backoff (1s, 2s, 4s).
- Events are dropped after max retries (logged as error).
- Runs in a background thread — never blocks the frame loop.
- Best-effort delivery: pipeline continues regardless of webhook status.
- Events are not re-sent after pipeline restart.

### Example Webhook Payload

```json
{"track_id": 1, "item_class": "plastic", "confidence": 0.8823, "frame_count": 12, "total_frames": 15, "is_certain": true, "timestamp": "2026-07-15T16:52:00+00:00", "hand_id": 1, "is_held_by_hand": true}
```

---

## Configuration

All tunable parameters live in [`config.yaml`](config.yaml). Key sections:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `model` | `weights` | `best.pt` | Model path |
| `model` | `confidence_threshold` | `0.25` | Min detection confidence |
| `trigger` | `motion_threshold` | `25.0` | Pixel diff threshold (0–255) |
| `trigger` | `area_fraction` | `0.005` | Frame area that must change to trigger |
| `buffer` | `active_window_size` | `30` | Max frames per detection window |
| `buffer` | `idle_timeout_frames` | `8` | Empty frames before early finalization |
| `buffer` | `min_frames_for_decision` | `5` | Min frames for valid vote |
| `voter` | `min_consensus_ratio` | `0.4` | Min agreement ratio for "certain" |
| `hand_tracking` | `backend` | `mediapipe` | Hand detection backend |
| `webhook` | `url` | `null` | Webhook endpoint (null = disabled) |

---

## Running Tests

```bash
# Run all tests (no GPU/camera required)
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=smartbin --cov-report=term-missing

# Lint check
ruff check smartbin/ tests/
```

Tests are designed to run **offline** — they mock the detector and don't require video input or GPU.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥8.0 | YOLO detection & tracking |
| `opencv-python` | ≥4.8 | Frame capture & display |
| `pyyaml` | ≥6.0 | Config loading |
| `numpy` | ≥1.24 | Numerical operations |
| `mediapipe` | ≥0.10 | Hand tracking (MediaPipe backend) |
| `pytest` | ≥7.0 | Testing (dev only) |
| `pytest-cov` | ≥4.0 | Coverage reporting (dev only) |
| `ruff` | ≥0.4 | Linting (dev only) |

---

## Troubleshooting

### Camera Connection Issues (Windows)

On Windows, the default MSMF backend sometimes fails for webcams. The pipeline automatically tries DirectShow first. If it still fails:

```bash
python main.py --source 1  # Try different camera index
```

### Model Class Validation Failure

If you see "MODEL CLASS MISMATCH DETECTED", your weights contain COCO classes instead of waste classes. Train a custom model:

```bash
python train_waste_model.py
```

Or override for testing: `python main.py --allow-generic-model`

### GPU Not Detected

Set the device explicitly in config.yaml:

```yaml
model:
  device: "cuda:0"  # or "cpu"
```

---

## Contributing

This is an **experimental prototype**. Contributions welcome, especially:
- **Edge hardware testing** — Jetson Orin Nano latency, memory usage
- **First-party bin images** — Labeled waste images from the target camera/hardware
- **Fine-tuned waste models** — Custom YOLO training on recycling/compost datasets
- **TensorRT implementation** — See stub at `smartbin/detector.py`
- **Test coverage** — More edge cases, integration tests with real video

---

## License

Proprietary — Cashcrow Technologies.

---

## Contact & Feedback

Built as an internship prototype. Questions or ideas? Open an issue or reach out to the maintainer.

**Happy detecting! 🚀**
