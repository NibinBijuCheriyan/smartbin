# Cashcrow Smartbin — AI Waste Detection Pipeline

Production-grade multi-frame waste detection pipeline for Cashcrow's Smartbin edge device. Uses YOLO + ByteTrack with sliding-window majority vote for stable, robust waste classification.

## Architecture

```
Camera/Video ──▶ Trigger Gate ──▶ State Machine ──▶ YOLO + ByteTrack ──▶ Majority Voter ──▶ Decision Hooks
                (frame diff)      (IDLE/ACTIVE)     (detection+tracking)  (consensus vote)    (JSONL, logging)
```

**Key design decisions:**
- **Trigger gate** keeps the detector dormant when the bin is idle (saves ~95% compute).
- **ByteTrack** (not DeepSORT) — no ReID features, minimal overhead, handles hand occlusion well.
- **Consensus-conditioned confidence** — averages only the frames that agreed with the winning label, so occluded/noisy frames don't dilute the signal.
- **Clean abstraction boundary** at the detector layer — swap YOLO for TensorRT with zero changes to trigger/vote logic.

## Project Structure

```
├── smartbin/                    # Main Python package
│   ├── config.py                # YAML config + CLI override merging
│   ├── trigger.py               # Motion/presence gate (frame differencing)
│   ├── state_machine.py         # IDLE ↔ ACTIVE lifecycle + frame buffer
│   ├── detector.py              # YOLO + ByteTrack wrapper (swappable)
│   ├── voter.py                 # Sliding-window majority vote
│   ├── decision.py              # Decision event dataclass + output hooks
│   └── pipeline.py              # Thin orchestrator wiring everything together
├── tests/                       # Test suite (runs without GPU/camera)
├── config.yaml                  # Default configuration (all tunable params)
├── main.py                      # CLI entry point
└── README.md
```

## Setup

### Prerequisites
- Python 3.9+
- (Optional) NVIDIA GPU with CUDA for accelerated inference

### Install

```bash
# Clone the repository
cd intern-cashcrow

# Install dependencies
pip install -r requirements.txt

# Or install as editable package
pip install -e ".[dev]"
```

### Model Weights

The pipeline ships with COCO-pretrained YOLO weights by default. On first run, Ultralytics will auto-download `yolo11n.pt` if it's not already present.

## Usage

### Run against a webcam

```bash
python main.py --source 0 --show
```

### Run against a video file

```bash
python main.py --source path/to/video.mp4 --show
```

### Custom model weights and confidence

```bash
python main.py --weights best.pt --confidence 0.5 --show
```

### Custom config file

```bash
python main.py --config my_config.yaml
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.yaml` | Path to YAML config file |
| `--source` | `0` (webcam) | Video source: integer for webcam, or file path |
| `--weights` | `yolo11n.pt` | Path to YOLO model weights |
| `--confidence` | `0.25` | Detection confidence threshold |
| `--show` | off | Show annotated live preview window |
| `--log-level` | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

CLI arguments override the corresponding values in the config file.

## Configuration Reference

All tunable parameters are in [`config.yaml`](config.yaml). Key sections:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `model` | `weights` | `yolo11n.pt` | Model weights path (swap-in point) |
| `model` | `confidence_threshold` | `0.25` | Min detection confidence |
| `trigger` | `motion_threshold` | `25.0` | Pixel diff threshold (0–255) |
| `trigger` | `area_fraction` | `0.005` | Frame area that must change |
| `buffer` | `active_window_size` | `30` | Max frames per detection window |
| `buffer` | `idle_timeout_frames` | `8` | Empty frames before early finalize |
| `buffer` | `min_frames_for_decision` | `5` | Min frames for a valid vote |
| `voter` | `min_consensus_ratio` | `0.4` | Min agreement for "certain" |
| `camera` | `fps_limit` | `15` | Max processing FPS |

## Swapping in a Fine-Tuned Waste Model

1. Train your waste classification model using Ultralytics YOLO (e.g., on a custom dataset with classes: plastic, paper, metal, glass, e-waste, organic).
2. Export the best weights (`best.pt`).
3. Update `config.yaml`:
   ```yaml
   model:
     weights: "path/to/best.pt"
   ```
   Or pass via CLI: `python main.py --weights path/to/best.pt`

**No other pipeline code changes are needed.** The detector reads class names from the model itself.

## Running Tests

```bash
# Run all tests (no GPU/camera required)
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=smartbin --cov-report=term-missing
```

## Decision Output

Each finalized decision is:
1. Logged at INFO level via Python's logging.
2. Appended as a JSON line to `decisions.jsonl` (configurable path).

**Example JSONL entry:**
```json
{"track_id":1,"item_class":"plastic","confidence":0.8823,"frame_count":12,"total_frames":15,"is_certain":true,"timestamp":"2026-07-13T15:30:00+00:00"}
```

## Extension Points

### TensorRT Deployment (Jetson)

The detector layer is abstracted behind `BaseDetector`. To deploy on Jetson:

1. Export the YOLO model to TensorRT:
   ```bash
   yolo export model=best.pt format=engine device=0
   ```
2. Implement `TensorRTDetector` in `smartbin/detector.py` (stub is already in place).
3. Update the `create_detector()` factory to route `.engine` files to the new class.

**Key principle:** Only `detector.py` changes. Trigger, state machine, voter, and hooks remain untouched.

---

## TODO / Next Steps

> These are planned enhancements — extension points are already in the code, but the implementations are not yet built.

- [ ] **TensorRT export & runtime** — Implement `TensorRTDetector` for Jetson Orin deployment. Use the `.engine` format with FP16 for optimal throughput.
- [ ] **Hand-detection-based triggering** — Train a lightweight hand detector to trigger the pipeline more precisely (vs. generic motion). This would reduce false triggers from shadows/lighting changes.
- [ ] **Hardware sensor triggers** — Integrate IR proximity sensor and/or weight sensor as trigger sources (subclass `BaseTrigger`).
- [ ] **Cloud/dashboard logging** — Implement `CloudSyncHook` to batch-upload decisions to a cloud dashboard for analytics and reporting.
- [ ] **MQTT bin actuation** — Implement `MqttHook` to publish decisions to an MQTT broker for physical bin sorting control.
- [ ] **Cross-session deduplication** — Detect when the same item is presented multiple times across separate IDLE→ACTIVE cycles.
- [ ] **DeepStream integration** — For multi-camera deployments, port the pipeline to NVIDIA DeepStream for hardware-accelerated video decoding and batched inference.

## License

Proprietary — Cashcrow Technologies.
