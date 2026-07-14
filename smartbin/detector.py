"""
Detection + tracking wrapper.

Encapsulates the YOLO model + ByteTrack tracker behind a clean interface
so that the rest of the pipeline (trigger, state machine, voter) never
touches Ultralytics directly.

This is the primary swap-in point for TensorRT: implement a TensorRTDetector
subclass that replaces the Ultralytics inference call with a TensorRT engine
call, returning the same Detection dataclass. No other module changes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from smartbin.config import ModelConfig, TrackerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """A single detection with its track ID from ByteTrack."""

    track_id: int  # -1 if tracker hasn't assigned an ID yet
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixel coordinates


# ---------------------------------------------------------------------------
# Abstract detector interface
# ---------------------------------------------------------------------------


class BaseDetector(ABC):
    """
    Abstract detector interface.

    Any detector (Ultralytics YOLO, TensorRT engine, ONNX Runtime, etc.)
    must implement `detect()` returning a list of Detection objects.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection + tracking on a single frame."""

    @abstractmethod
    def reset_tracker(self) -> None:
        """Reset the tracker state (called on IDLE → ACTIVE transition)."""

    def close(self) -> None:
        """Release resources. Default: no-op."""


# ---------------------------------------------------------------------------
# Ultralytics YOLO + ByteTrack implementation
# ---------------------------------------------------------------------------


class YOLODetector(BaseDetector):
    """
    YOLO detector using the Ultralytics library with built-in ByteTrack.

    The model is loaded once at init. Each call to `detect()` runs
    `model.track()` with `persist=True` to maintain track IDs across frames.

    ByteTrack is chosen over BoT-SORT because:
    - It's faster (no ReID feature extraction).
    - It handles the smartbin use case well (one item, brief occlusion by hand).
    - BoT-SORT's appearance features don't help when items are partially
      covered by a hand at close range.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        tracker_config: TrackerConfig,
    ) -> None:
        self._conf_threshold = model_config.confidence_threshold
        self._tracker_type = f"{tracker_config.type}.yaml"
        self._model = None
        self._model_config = model_config
        self._tracker_config = tracker_config

        # Lazy loading — model is loaded on first detect() call.
        # This allows tests to instantiate the detector without needing
        # actual model weights.
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load the YOLO model on first use."""
        if self._loaded:
            return

        from ultralytics import YOLO

        weights = self._model_config.weights
        if not Path(weights).exists() and not weights.startswith("yolo"):
            raise FileNotFoundError(
                f"Model weights not found: {weights}. "
                f"Download a pretrained model or provide a valid path."
            )

        logger.info("Loading YOLO model: %s", weights)
        self._model = YOLO(weights)

        # Set device
        device = self._model_config.device
        if device != "auto":
            self._model.to(device)

        self._loaded = True
        logger.info(
            "Model loaded (device=%s, tracker=%s)",
            device,
            self._tracker_type,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO detection + ByteTrack tracking on a single frame.

        Returns a list of Detection objects, one per detected item.
        Items without a track ID (first frame, tracker warm-up) are
        included with track_id=-1.
        """
        self._ensure_loaded()

        results = self._model.track(
            source=frame,
            persist=True,
            tracker=self._tracker_type,
            conf=self._conf_threshold,
            verbose=False,  # Suppress Ultralytics' own logging
        )

        detections: List[Detection] = []

        if not results or len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Extract arrays
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        bboxes = boxes.xyxy.cpu().numpy()

        # Track IDs may be None on the first frame or when ByteTrack
        # hasn't assigned an ID yet.
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.full(len(boxes), -1, dtype=int)

        names = result.names  # {class_id: class_name} mapping

        for i in range(len(boxes)):
            det = Detection(
                track_id=int(track_ids[i]),
                class_id=int(classes[i]),
                class_name=names.get(int(classes[i]), f"class_{classes[i]}"),
                confidence=float(confs[i]),
                bbox=tuple(bboxes[i].tolist()),
            )
            detections.append(det)

        return detections

    def reset_tracker(self) -> None:
        """
        Reset ByteTrack state.

        Called when the state machine returns to IDLE so that the next
        active window starts with fresh track IDs (no carry-over from
        the previous item).
        """
        if self._model is not None and hasattr(self._model, "predictor"):
            predictor = self._model.predictor
            if predictor is not None and hasattr(predictor, "trackers"):
                for tracker in predictor.trackers:
                    tracker.reset()
                logger.debug("Tracker state reset")

    def close(self) -> None:
        """Release model resources."""
        self._model = None
        self._loaded = False


# ---------------------------------------------------------------------------
# TensorRT detector stub — extension point
# ---------------------------------------------------------------------------

# TODO: Implement TensorRTDetector for production Jetson deployment.
#
# class TensorRTDetector(BaseDetector):
#     """
#     TensorRT-accelerated detector for Jetson deployment.
#
#     Uses a .engine file exported from the YOLO model. The tracking
#     logic (ByteTrack) runs separately via the `supervision` or
#     standalone `bytetrack` Python package, since Ultralytics' built-in
#     tracker is tied to its own inference pipeline.
#
#     Swap-in steps:
#     1. Export YOLO model: `yolo export model=best.pt format=engine`
#     2. Set config.model.weights to the .engine file path.
#     3. Update detector factory to return TensorRTDetector when weights
#        end with '.engine'.
#     """
#
#     def __init__(self, model_config, tracker_config):
#         # Load TensorRT engine
#         # Initialize standalone ByteTrack
#         pass
#
#     def detect(self, frame):
#         # Run TensorRT inference
#         # Run ByteTrack on detections
#         # Return List[Detection]
#         pass
#
#     def reset_tracker(self):
#         pass


def create_detector(
    model_config: ModelConfig,
    tracker_config: TrackerConfig,
) -> BaseDetector:
    """Factory function — returns the appropriate detector for the config."""
    # TODO: Check for .engine extension and return TensorRTDetector
    return YOLODetector(model_config, tracker_config)
