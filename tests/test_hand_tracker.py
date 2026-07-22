"""
Unit tests for hand tracking and hand-held object association module.
"""

import numpy as np
import pytest
from smartbin.detector import Detection
from smartbin.hand_tracker import (
    HandDetection,
    HandTracker,
    associate_hands_and_objects,
    get_hand_roi,
)


def test_hand_tracker_init():
    tracker = HandTracker()
    assert tracker.confidence_threshold == 0.3
    assert tracker.max_distance_px == 120.0


def test_hand_tracker_detect_blank_frame():
    tracker = HandTracker()
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    hands = tracker.detect_and_track(frame)
    assert isinstance(hands, list)
    assert len(hands) == 0


def test_hand_tracker_detect_synthetic_hand():
    tracker = HandTracker()
    frame = np.zeros((320, 320, 3), dtype=np.uint8)

    # Draw synthetic skin-tone rectangle in HSV skin range
    # In BGR: roughly (120, 150, 200) -> orange/skin tone
    frame[50:150, 50:150] = (120, 150, 200)

    hands = tracker.detect_and_track(frame)
    assert len(hands) == 1
    assert hands[0].hand_id == 1
    assert hands[0].bbox[0] < hands[0].bbox[2]
    assert hands[0].bbox[1] < hands[0].bbox[3]


def test_associate_hands_and_objects():
    hands = [
        HandDetection(
            hand_id=1,
            confidence=0.9,
            bbox=(50.0, 50.0, 150.0, 150.0),
            center=(100.0, 100.0),
        )
    ]

    # Object overlapping with hand
    det_held = Detection(
        track_id=10,
        class_id=0,
        class_name="plastic",
        confidence=0.95,
        bbox=(80.0, 80.0, 120.0, 120.0),
    )

    # Object far from hand
    det_far = Detection(
        track_id=11,
        class_id=1,
        class_name="paper",
        confidence=0.88,
        bbox=(400.0, 400.0, 450.0, 450.0),
    )

    updated = associate_hands_and_objects(hands, [det_held, det_far], max_dist_px=100.0)

    assert len(updated) == 2
    assert updated[0].hand_id == 1
    assert updated[0].is_held_by_hand is True

    assert updated[1].hand_id is None
    assert updated[1].is_held_by_hand is False


def test_get_hand_roi():
    hands = [
        HandDetection(
            hand_id=1,
            confidence=0.9,
            bbox=(100.0, 100.0, 200.0, 200.0),
            center=(150.0, 150.0),
        )
    ]

    roi = get_hand_roi((480, 640, 3), hands, padding_factor=1.4)
    assert roi is not None
    x1, y1, x2, y2 = roi
    assert 0 <= x1 < 100
    assert 0 <= y1 < 100
    assert 200 < x2 <= 640
    assert 200 < y2 <= 480
