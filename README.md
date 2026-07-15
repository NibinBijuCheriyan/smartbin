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
                                                                              │  (JSONL + Logs)  │
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
- Tests runnable without GPU or camera

###  Known Limitations
- **Generic YOLO model** — Uses COCO pretrained weights. Does not distinguish waste types (plastic, paper, metal, organic). Fine-tuned model required for real-world deployment.
- **No TensorRT support yet** — Only Ultralytics YOLO via PyTorch. TensorRT stub present for future Jetson optimization.
- **Limited edge testing** — Developed locally; not yet validated on actual Jetson hardware.
- **No fine-tuned waste dataset** — COCO classes include "person", "backpack", etc. Not optimized for recycling/compost sorting.
- **Minimal production hardening** — No cloud sync, MQTT actuation, or multi-bin orchestration.

---

## Project Structure

```
smartbin/
├── smartbin/                       # Main Python package
│   ├── config.py                   # YAML config loading + CLI merge
│   ├── trigger.py                  # Motion gate (frame differencing)
│   ├── state_machine.py            # IDLE ↔ ACTIVE lifecycle + buffering
│   ├── detector.py                 # YOLO + ByteTrack wrapper (abstracted)
│   ├── voter.py                    # Sliding-window majority vote
│   ├── decision.py                 # Decision event dataclass + output hooks
│   └── pipeline.py                 # Orchestrator (thin wiring layer)
├── tests/                          # Test suite (runs offline)
├── config.yaml                     # Default configuration (all tunable)
├── main.py                         # CLI entry point
├── requirements.txt                # Dependencies
└── README.md
```

### Runtime Flow

1. **Main loop** (`pipeline.py`) reads frames from webcam or video file at configurable FPS.
2. **IDLE state** — Motion trigger checks each frame. Dormant mode keeps compute cost near zero.
3. **ACTIVE state (on trigger)** — YOLO detector runs on each frame. ByteTrack assigns/maintains object IDs.
4. **State machine** buffers detections until window fills or no motion for N frames.
5. **Voter** aggregates per-track class predictions via majority vote + consensus confidence.
6. **Decision hooks** output to logging and JSONL, then return to IDLE.

---

## Setup

### Prerequisites
- **Python 3.9+**
- (Optional) NVIDIA GPU with CUDA for faster inference

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

The pipeline ships with **COCO-pretrained YOLO11-nano** by default. On first run, Ultralytics will auto-download `yolo11n.pt` (~13 MB) if not present.

**Warning:** COCO classes do not include waste categories. For real recycling/compost sorting, you must fine-tune a custom model.

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

### Custom Config

```bash
python main.py --config my_config.yaml
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.yaml` | Path to YAML config file |
| `--source` | `0` (webcam) | Video source: integer for webcam ID, or file path |
| `--weights` | `yolo11n.pt` | Path to YOLO model weights |
| `--confidence` | `0.25` | Detection confidence threshold (0–1) |
| `--show` | off | Display annotated live preview window |
| `--log-level` | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

CLI arguments **override** config file values.

---

## Configuration

All tunable parameters live in [`config.yaml`](config.yaml). Key sections:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `model` | `weights` | `yolo11n.pt` | Model path (YOLO only, TensorRT stub present) |
| `model` | `confidence_threshold` | `0.25` | Min detection confidence |
| `trigger` | `motion_threshold` | `25.0` | Pixel diff threshold (0–255) |
| `trigger` | `area_fraction` | `0.005` | Frame area that must change to trigger |
| `trigger` | `roi` | `null` | Optional region of interest [x1, y1, x2, y2] |
| `buffer` | `active_window_size` | `30` | Max frames per detection window |
| `buffer` | `idle_timeout_frames` | `8` | Empty frames before early finalization |
| `buffer` | `min_frames_for_decision` | `5` | Min frames for valid vote |
| `voter` | `min_consensus_ratio` | `0.4` | Min agreement ratio for "certain" |
| `camera` | `fps_limit` | `15` | Max processing FPS |
| `logging` | `level` | `INFO` | Log verbosity |
| `logging` | `decision_log` | `decisions.jsonl` | Output JSONL path |

---

## Decision Output

Each finalized decision is:
1. **Logged** at INFO level via Python's `logging` module.
2. **Appended** as a JSON line to the configured JSONL file (default: `decisions.jsonl`).

### Example JSONL Entry

```json
{"track_id": 1, "item_class": "backpack", "confidence": 0.8823, "frame_count": 12, "total_frames": 15, "is_certain": true, "timestamp": "2026-07-15T16:52:00+00:00"}
```

Fields:
- `track_id` — Unique ID for this tracked object (ByteTrack)
- `item_class` — Winning class from majority vote (e.g., "person", "backpack", "bottle")
- `confidence` — Mean confidence of frames that predicted the winning class
- `frame_count` — Frames that voted for the winning class
- `total_frames` — Total frames this track was observed
- `is_certain` — True if `frame_count / total_frames >= min_consensus_ratio`
- `timestamp` — ISO 8601 decision timestamp

---

## Running Tests

```bash
# Run all tests (no GPU/camera required)
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=smartbin --cov-report=term-missing
```

Tests are designed to run **offline** — they mock the detector and don't require video input or GPU.

---

## Customization & Extension

### Fine-Tuning a Waste Model

To deploy with custom waste classes (plastic, paper, metal, glass, organic):

1. **Collect & label** waste images (or use an existing dataset like TACO).
2. **Train** a custom YOLO model:
   ```bash
   yolo detect train data=path/to/dataset.yaml epochs=100 imgsz=640
   ```
3. **Export** the best weights:
   ```bash
   cp runs/detect/train/weights/best.pt path/to/best.pt
   ```
4. **Update** `config.yaml`:
   ```yaml
   model:
     weights: "path/to/best.pt"
   ```
   Or pass via CLI: `python main.py --weights path/to/best.pt`

**No other pipeline code changes needed.** The detector reads class names directly from the model.

### TensorRT Deployment (Jetson) — Future

The detector layer is abstracted behind `BaseDetector`. To enable TensorRT:

1. **Export** YOLO to TensorRT:
   ```bash
   yolo export model=best.pt format=engine device=0
   ```
2. **Implement** `TensorRTDetector` in `smartbin/detector.py` (stub at line 213).
3. **Update** the `create_detector()` factory to route `.engine` files.

**Key principle:** Only `detector.py` changes. Trigger, state machine, voter, and hooks remain untouched.

---

## Planned Enhancements

> These are aspirational extensions — hooks and stubs are already in the code.

- [ ] **TensorRT export & runtime** — Implement `TensorRTDetector` for Jetson Orin Nano. Use FP16 for optimal throughput.
- [ ] **Hand-detection-based triggering** — Train a lightweight hand detector to trigger more precisely (vs. generic motion). Reduces false positives from shadows/lighting changes.
- [ ] **Hardware sensor triggers** — Integrate IR proximity or weight sensors (subclass `BaseTrigger`).
- [ ] **Cloud/dashboard logging** — Implement `CloudSyncHook` to batch-upload decisions for analytics.
- [ ] **MQTT bin actuation** — Implement `MqttHook` to publish decisions to MQTT broker for physical sorting.
- [ ] **Cross-session deduplication** — Detect when the same item is presented multiple times across IDLE↔ACTIVE cycles.
- [ ] **DeepStream integration** — For multi-camera setups on hardware-accelerated video decoding.

---

## Troubleshooting

### Camera Connection Issues (Windows)

On Windows, the default MSMF backend sometimes fails for webcams. The pipeline automatically tries DirectShow first. If it still fails:

```bash
python main.py --source 1  # Try different camera index
```

### GPU Not Detected

Set the device explicitly:

```bash
python main.py --device cuda:0  # or "cpu" to force CPU inference
```

### Model Download Hangs

If `yolo11n.pt` download stalls, download manually:

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
```

Or pre-download via CLI:
```bash
yolo detect predict model=yolo11n.pt source="test.jpg"
```

---

## Questions to Explore

- **How do I fine-tune the model for plastic/paper/metal/glass waste?** See [Fine-Tuning a Waste Model](#fine-tuning-a-waste-model).
- **What's the latency on a Jetson Orin Nano?** Not yet tested. TensorRT implementation will help. Contribute results!
- **How do I add a weight sensor as a trigger?** Subclass `BaseTrigger` in `smartbin/trigger.py` and set `config.trigger.method`.
- **Can I run on multiple cameras?** Not yet. DeepStream integration (planned) would enable this.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥8.0 | YOLO detection & tracking |
| `opencv-python` | ≥4.8 | Frame capture & display |
| `pyyaml` | ≥6.0 | Config loading |
| `numpy` | ≥1.24 | Numerical operations |
| `pytest` | ≥7.0 | Testing (dev only) |
| `pytest-cov` | ≥4.0 | Coverage reporting (dev only) |

---

## Contributing

This is an **experimental prototype**. Contributions welcome, especially:
- **Edge hardware testing** — Jetson Orin Nano latency, memory usage
- **Fine-tuned waste models** — Custom YOLO training on recycling/compost datasets
- **TensorRT implementation** — See stub at `smartbin/detector.py:213`
- **Test coverage** — More edge cases, integration tests with real video

---

## License

Proprietary — Cashcrow Technologies.

---

## Contact & Feedback

Built as an internship prototype. Questions or ideas? Open an issue or reach out to the maintainer.

**Happy detecting! **
