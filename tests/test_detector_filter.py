"""Tests for detector post-processing filters."""

from __future__ import annotations

from smartbin.config import ModelConfig, TrackerConfig
from smartbin.detector import Detection, YOLODetector


def test_detector_drops_classes_outside_allowlist():
    detector = YOLODetector(
        ModelConfig(allowed_classes=["plastic", "paper"]),
        TrackerConfig(),
    )

    person = Detection(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=(10, 10, 100, 100),
    )
    plastic = Detection(
        track_id=2,
        class_id=1,
        class_name="plastic",
        confidence=0.8,
        bbox=(10, 10, 100, 100),
    )

    frame_shape = (480, 640, 3)

    assert detector._is_valid_detection(person, frame_shape) is False
    assert detector._is_valid_detection(plastic, frame_shape) is True


def test_detector_drops_implausible_box_sizes():
    detector = YOLODetector(
        ModelConfig(
            allowed_classes=["plastic"],
            min_box_area_fraction=0.01,
            max_box_area_fraction=0.5,
        ),
        TrackerConfig(),
    )

    tiny = Detection(
        track_id=1,
        class_id=0,
        class_name="plastic",
        confidence=0.9,
        bbox=(10, 10, 20, 20),
    )
    huge = Detection(
        track_id=2,
        class_id=0,
        class_name="plastic",
        confidence=0.9,
        bbox=(0, 0, 640, 480),
    )

    frame_shape = (480, 640, 3)

    assert detector._is_valid_detection(tiny, frame_shape) is False
    assert detector._is_valid_detection(huge, frame_shape) is False


def test_detector_allows_classes_in_class_agnostic_mode():
    detector = YOLODetector(
        ModelConfig(allowed_classes=["plastic", "paper"], class_agnostic=True),
        TrackerConfig(),
    )

    person = Detection(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=(10, 10, 100, 100),
    )
    plastic = Detection(
        track_id=2,
        class_id=1,
        class_name="plastic",
        confidence=0.8,
        bbox=(10, 10, 100, 100),
    )

    frame_shape = (480, 640, 3)

    assert detector._is_valid_detection(person, frame_shape) is True
    assert detector._is_valid_detection(plastic, frame_shape) is True


def test_detector_per_class_confidence_threshold():
    """Dict confidence_threshold computes correct min for YOLO conf= param."""
    thresholds = {"plastic": 0.40, "paper": 0.40, "other": 0.55}
    detector = YOLODetector(
        ModelConfig(
            allowed_classes=["plastic", "paper", "other"],
            confidence_threshold=thresholds,
        ),
        TrackerConfig(),
    )
    # The YOLO conf= parameter should use the minimum threshold
    assert detector._conf_threshold == 0.40


def test_detector_per_class_threshold_uses_min_for_scalar():
    """Scalar confidence_threshold is used directly as conf= param."""
    detector = YOLODetector(
        ModelConfig(
            allowed_classes=["plastic"],
            confidence_threshold=0.35,
        ),
        TrackerConfig(),
    )
    assert detector._conf_threshold == 0.35