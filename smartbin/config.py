"""
Configuration loader for the Smartbin pipeline.

Loads parameters from a YAML config file and merges CLI overrides.
CLI arguments always take precedence over file values.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dataclasses — frozen after construction for safety
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    weights: str = "yolo11n.pt"
    confidence_threshold: float = 0.25
    device: str = "auto"


@dataclass(frozen=True)
class TriggerConfig:
    method: str = "frame_diff"
    motion_threshold: float = 25.0
    area_fraction: float = 0.005
    roi: Optional[List[int]] = None  # [x1, y1, x2, y2]
    background_alpha: float = 0.05


@dataclass(frozen=True)
class BufferConfig:
    active_window_size: int = 30
    idle_timeout_frames: int = 8
    min_frames_for_decision: int = 5


@dataclass(frozen=True)
class TrackerConfig:
    type: str = "bytetrack"
    track_buffer: int = 30
    match_thresh: float = 0.8


@dataclass(frozen=True)
class VoterConfig:
    min_consensus_ratio: float = 0.4


@dataclass(frozen=True)
class CameraConfig:
    source: Union[int, str] = 0
    fps_limit: int = 15


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    output_file: Optional[str] = None
    decision_log: str = "decisions.jsonl"


@dataclass(frozen=True)
class DisplayConfig:
    show: bool = False


@dataclass(frozen=True)
class HandTrackingConfig:
    enabled: bool = False
    confidence_threshold: float = 0.3
    max_hand_distance_px: float = 150.0
    roi_padding_factor: float = 1.4
    roi_crop_enabled: bool = True


@dataclass(frozen=True)
class SmartbinConfig:
    """Top-level configuration for the entire pipeline."""

    model: ModelConfig = field(default_factory=ModelConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    voter: VoterConfig = field(default_factory=VoterConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    hand_tracking: HandTrackingConfig = field(default_factory=HandTrackingConfig)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Union[str, Path]) -> dict:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(data)}")
    return data


def _build_section(cls, raw: dict, section_key: str):
    """Build a frozen dataclass from the corresponding YAML section."""
    section_data = raw.get(section_key, {})
    if section_data is None:
        section_data = {}
    # Only pass keys the dataclass actually expects
    valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in section_data.items() if k in valid_keys}
    return cls(**filtered)


def load_config_from_yaml(path: Union[str, Path]) -> SmartbinConfig:
    """Parse a YAML config file into a validated SmartbinConfig."""
    raw = _load_yaml(path)
    return SmartbinConfig(
        model=_build_section(ModelConfig, raw, "model"),
        trigger=_build_section(TriggerConfig, raw, "trigger"),
        buffer=_build_section(BufferConfig, raw, "buffer"),
        tracker=_build_section(TrackerConfig, raw, "tracker"),
        voter=_build_section(VoterConfig, raw, "voter"),
        camera=_build_section(CameraConfig, raw, "camera"),
        logging=_build_section(LoggingConfig, raw, "logging"),
        display=_build_section(DisplayConfig, raw, "display"),
        hand_tracking=_build_section(HandTrackingConfig, raw, "hand_tracking"),
    )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser. CLI args override config-file values."""
    parser = argparse.ArgumentParser(
        prog="smartbin",
        description="Cashcrow Smartbin — AI waste detection pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: integer for webcam index, or path to video file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to YOLO model weights (overrides config)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Detection confidence threshold (overrides config)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=None,
        help="Show annotated live preview window",
    )
    parser.add_argument(
        "--track-hands",
        action="store_true",
        default=None,
        help="Enable hand tracking and hand-held object detection",
    )
    parser.add_argument(
        "--hand-roi",
        action="store_true",
        default=None,
        help="Enable hand ROI cropping for efficient frame detection",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (overrides config)",
    )
    return parser


def _parse_source(source_str: str) -> Union[int, str]:
    """Convert a source string to int (webcam index) or keep as path."""
    try:
        return int(source_str)
    except ValueError:
        return source_str


def load_config(argv: Optional[list] = None) -> SmartbinConfig:
    """
    Load configuration by merging YAML file with CLI overrides.

    Priority: CLI args > YAML file > dataclass defaults.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Load base config from YAML (fall back to defaults if file missing)
    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_config_from_yaml(config_path)
        logger.info("Loaded config from %s", config_path)
    else:
        logger.warning(
            "Config file %s not found, using defaults", config_path
        )
        cfg = SmartbinConfig()

    hand_enabled = cfg.hand_tracking.enabled
    if args.track_hands is not None and args.track_hands:
        hand_enabled = True

    roi_enabled = cfg.hand_tracking.roi_crop_enabled
    if args.hand_roi is not None and args.hand_roi:
        roi_enabled = True

    ht_cfg = HandTrackingConfig(
        enabled=hand_enabled,
        confidence_threshold=cfg.hand_tracking.confidence_threshold,
        max_hand_distance_px=cfg.hand_tracking.max_hand_distance_px,
        roi_padding_factor=cfg.hand_tracking.roi_padding_factor,
        roi_crop_enabled=roi_enabled,
    )

    # Apply CLI overrides
    model_weights = args.weights if args.weights is not None else cfg.model.weights
    model_conf = args.confidence if args.confidence is not None else cfg.model.confidence_threshold
    source = _parse_source(args.source) if args.source is not None else cfg.camera.source
    show = args.show if args.show is not None else cfg.display.show
    log_lvl = args.log_level if args.log_level is not None else cfg.logging.level

    cfg = SmartbinConfig(
        model=ModelConfig(
            weights=model_weights,
            confidence_threshold=model_conf,
            device=cfg.model.device,
        ),
        trigger=cfg.trigger,
        buffer=cfg.buffer,
        tracker=cfg.tracker,
        voter=cfg.voter,
        camera=CameraConfig(source=source, fps_limit=cfg.camera.fps_limit),
        logging=LoggingConfig(
            level=log_lvl,
            output_file=cfg.logging.output_file,
            decision_log=cfg.logging.decision_log,
        ),
        display=DisplayConfig(show=show),
        hand_tracking=ht_cfg,
    )

    return cfg


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(cfg: LoggingConfig) -> None:
    """Configure Python logging from the LoggingConfig."""
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if cfg.output_file:
        os.makedirs(os.path.dirname(cfg.output_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(cfg.output_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,  # Override any existing config
    )
