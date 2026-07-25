"""
Unit tests for hand tracking and hand-held object association module.

Tests cover:
- SkinColorHandTracker (legacy): basic detection with synthetic skin-tone frames.
- MediaPipeHandTracker: initialization and interface compliance.
- Spatial association: hand-object linking.
- ROI computation.
- Trigger thrashing: verify noisy hand detections don't cause rapid IDLE↔ACTIVE flapping.
- Factory function: create_hand_tracker() backend selection.
"""

import numpy as np
import pytest
from smartbin.detector import Detection
from smartbin.hand_tracker import (
    HandDetection,
    MediaPipeHandTracker,
    SkinColorHandTracker,
    associate_hands_and_objects,
    create_hand_tracker,
    get_hand_roi,
)


# ---------------------------------------------------------------------------
# SkinColorHandTracker tests (legacy backend)
# ---------------------------------------------------------------------------


def test_skin_color_tracker_init():
    tracker = SkinColorHandTracker()
    assert tracker.confidence_threshold == 0.3
    assert tracker.max_distance_px == 120.0


def test_skin_color_tracker_detect_blank_frame():
    tracker = SkinColorHandTracker()
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    hands = tracker.detect_and_track(frame)
    assert isinstance(hands, list)
    assert len(hands) == 0


def test_skin_color_tracker_detect_synthetic_hand():
    tracker = SkinColorHandTracker()
    frame = np.zeros((320, 320, 3), dtype=np.uint8)

    # Draw synthetic skin-tone rectangle in HSV skin range
    # In BGR: roughly (120, 150, 200) -> orange/skin tone
    frame[50:150, 50:150] = (120, 150, 200)

    hands = tracker.detect_and_track(frame)
    assert len(hands) == 1
    assert hands[0].hand_id == 1
    assert hands[0].bbox[0] < hands[0].bbox[2]
    assert hands[0].bbox[1] < hands[0].bbox[3]


# ---------------------------------------------------------------------------
# MediaPipeHandTracker tests
# ---------------------------------------------------------------------------


def test_mediapipe_tracker_init():
    """MediaPipeHandTracker initializes without errors (lazy-loads MediaPipe)."""
    try:
        tracker = MediaPipeHandTracker()
        assert tracker.confidence_threshold == 0.3
    except ImportError:
        pytest.skip("MediaPipe not installed")


def test_mediapipe_tracker_detect_blank_frame():
    """MediaPipeHandTracker returns empty list for blank frame."""
    try:
        tracker = MediaPipeHandTracker()
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        hands = tracker.detect_and_track(frame)
        assert isinstance(hands, list)
        assert len(hands) == 0
    except ImportError:
        pytest.skip("MediaPipe not installed")


def test_mediapipe_tracker_returns_hand_detections():
    """MediaPipeHandTracker returns HandDetection objects with valid fields."""
    try:
        tracker = MediaPipeHandTracker()
        # A blank frame should return no hands
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        hands = tracker.detect_and_track(frame)
        assert isinstance(hands, list)
        for hand in hands:
            assert isinstance(hand, HandDetection)
            assert hand.hand_id > 0
            assert len(hand.bbox) == 4
            assert len(hand.center) == 2
    except ImportError:
        pytest.skip("MediaPipe not installed")


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


def test_create_hand_tracker_skin_color():
    tracker = create_hand_tracker(backend="skin_color")
    assert isinstance(tracker, SkinColorHandTracker)


def test_create_hand_tracker_mediapipe():
    try:
        tracker = create_hand_tracker(backend="mediapipe")
        assert isinstance(tracker, MediaPipeHandTracker)
    except ImportError:
        pytest.skip("MediaPipe not installed")


def test_create_hand_tracker_invalid_backend():
    with pytest.raises(ValueError, match="Unknown hand tracker backend"):
        create_hand_tracker(backend="nonexistent")


# ---------------------------------------------------------------------------
# Association and ROI tests (backend-independent)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Trigger thrashing test
# ---------------------------------------------------------------------------


def test_hand_detection_no_trigger_thrashing():
    """
    Verify that noisy hand detections near the ROI boundary don't cause
    rapid IDLE↔ACTIVE flapping.

    Simulates a sequence where hand detection alternates between detected
    and not-detected on every frame (worst case for thrashing). Verifies
    that the tracker's centroid persistence (max_disappeared) smooths this out.
    """
    tracker = SkinColorHandTracker(max_disappeared=5)

    # Create frames: alternating skin-tone patch present / absent
    frame_with_hand = np.zeros((320, 320, 3), dtype=np.uint8)
    frame_with_hand[50:150, 50:150] = (120, 150, 200)  # Skin tone

    frame_without_hand = np.zeros((320, 320, 3), dtype=np.uint8)

    state_changes = 0
    had_hands = False

    for i in range(20):
        # Alternate every frame (worst case)
        if i % 2 == 0:
            hands = tracker.detect_and_track(frame_with_hand)
        else:
            hands = tracker.detect_and_track(frame_without_hand)

        now_has_hands = len(hands) > 0

        if now_has_hands != had_hands:
            state_changes += 1
            had_hands = now_has_hands

    # With max_disappeared=5, the tracker should maintain the hand track
    # through the 1-frame gaps, resulting in far fewer state transitions
    # than the 20 alternations would cause without persistence.
    # Without persistence: up to 20 transitions. With persistence: ≤ 4.
    assert state_changes <= 4, (
        f"Trigger thrashing detected: {state_changes} state changes in 20 frames. "
        f"Centroid tracker should smooth out single-frame detection gaps."
    )


def test_tracker_reset_clears_state():
    """Verify reset() clears all tracking state."""
    tracker = SkinColorHandTracker()
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    frame[50:150, 50:150] = (120, 150, 200)

    hands = tracker.detect_and_track(frame)
    assert len(hands) > 0

    tracker.reset()

    # After reset, next detection should start with fresh IDs
    hands2 = tracker.detect_and_track(frame)
    if len(hands2) > 0:
        assert hands2[0].hand_id == 1  # Fresh ID after reset
