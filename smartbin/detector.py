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
from typing import List, Optional, Tuple

import numpy as np

from smartbin.config import ModelConfig, TrackerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """A single detection with its track ID from ByteTrack and optional hand tracking data."""

    track_id: int  # -1 if tracker hasn't assigned an ID yet
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixel coordinates
    hand_id: Optional[int] = None
    is_held_by_hand: bool = False
    raw_yolo_class: Optional[str] = None
    raw_yolo_conf: Optional[float] = None
    is_refined: bool = False

    def with_refinement(
        self,
        new_class_name: str,
        new_confidence: float,
        new_class_id: Optional[int] = None,
    ) -> Detection:
        """Return a new Detection object with refined class label, confidence, and class_id."""
        updated_class_id = new_class_id if new_class_id is not None else self.class_id
        return Detection(
            track_id=self.track_id,
            class_id=updated_class_id,
            class_name=new_class_name,
            confidence=new_confidence,
            bbox=self.bbox,
            hand_id=self.hand_id,
            is_held_by_hand=self.is_held_by_hand,
            raw_yolo_class=self.raw_yolo_class or self.class_name,
            raw_yolo_conf=self.raw_yolo_conf if self.raw_yolo_conf is not None else self.confidence,
            is_refined=True,
        )


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
    def detect(
        self,
        frame: np.ndarray,
        crop_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Detection]:
        """Run detection + tracking on a single frame or cropped ROI."""

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
        self._allowed_classes = (
            {name.lower() for name in model_config.allowed_classes}
            if model_config.allowed_classes
            else None
        )

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

    def detect(
        self,
        frame: np.ndarray,
        crop_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Detection]:
        """
        Run YOLO detection + ByteTrack tracking on a single frame or cropped ROI.

        Returns a list of Detection objects, one per detected item.
        Items without a track ID (first frame, tracker warm-up) are
        included with track_id=-1.
        """
        self._ensure_loaded()

        input_frame = frame
        offset_x, offset_y = 0, 0

        if crop_roi is not None:
            x1, y1, x2, y2 = crop_roi
            if x2 > x1 and y2 > y1:
                input_frame = frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1

        results = self._model.track(
            source=input_frame,
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
            box = bboxes[i].tolist()
            # Map back to full frame coordinates if cropped
            full_box = (
                box[0] + offset_x,
                box[1] + offset_y,
                box[2] + offset_x,
                box[3] + offset_y,
            )
            class_str = names.get(int(classes[i]), f"class_{classes[i]}")
            conf_val = float(confs[i])
            det = Detection(
                track_id=int(track_ids[i]),
                class_id=int(classes[i]),
                class_name=class_str,
                confidence=conf_val,
                bbox=full_box,
                raw_yolo_class=class_str,
                raw_yolo_conf=conf_val,
            )
            if self._is_valid_detection(det, frame.shape):
                detections.append(det)

        return detections

    def _is_valid_detection(
        self,
        det: Detection,
        frame_shape: Tuple[int, int, int],
    ) -> bool:
        """Filter non-waste classes and implausible boxes before voting."""
        class_name = det.class_name.lower()
        if self._allowed_classes is not None and class_name not in self._allowed_classes:
            logger.debug("Dropping class outside allowlist: %s", det.class_name)
            return False

        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = det.bbox
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        if box_w <= 0.0 or box_h <= 0.0:
            return False

        area_fraction = (box_w * box_h) / float(frame_w * frame_h)
        if area_fraction < self._model_config.min_box_area_fraction:
            return False
        if area_fraction > self._model_config.max_box_area_fraction:
            return False

        aspect_ratio = box_w / box_h
        return (
            self._model_config.min_box_aspect_ratio
            <= aspect_ratio
            <= self._model_config.max_box_aspect_ratio
        )

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
# TensorRT detector implementation
# ---------------------------------------------------------------------------


class TensorRTDetector(YOLODetector):
    """
    TensorRT-accelerated detector for Jetson deployment.

    Uses a .engine file exported from the YOLO model. Leverages Ultralytics
    native TensorRT inference runtime to execute the engine on Nvidia Jetson GPUs
    and run ByteTrack for object tracking.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        tracker_config: TrackerConfig,
    ) -> None:
        super().__init__(model_config, tracker_config)
        logger.info("Initializing TensorRTDetector with engine weights: %s", model_config.weights)


def create_detector(
    model_config: ModelConfig,
    tracker_config: TrackerConfig,
) -> BaseDetector:
    """Factory function — returns the appropriate detector for the config."""
    weights = str(model_config.weights)
    if weights.endswith(".engine"):
        return TensorRTDetector(model_config, tracker_config)
    return YOLODetector(model_config, tracker_config)

