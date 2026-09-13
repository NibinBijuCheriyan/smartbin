"""
Unit tests for multi-object prioritizer, temporal voting, and classifiers.
"""

import numpy as np
import pytest

from smartbin_v2.inference.contamination_detector import ContaminationDetector
from smartbin_v2.inference.engine import DetectionResult
from smartbin_v2.inference.material_classifier import MaterialClassifier
from smartbin_v2.inference.prioritizer import MultiObjectPrioritizer
from smartbin_v2.inference.temporal_voter import TemporalVotingBuffer


def test_multi_object_prioritizer():
    """Verify prioritizer selects center, high-area, confident item."""
    prioritizer = MultiObjectPrioritizer()
    frame_w, frame_h = 640, 640

    # Object 1: Medium in corner
    det1 = DetectionResult(
        class_id=0,
        class_name="aluminium_can",
        confidence=0.70,
        bbox=(20.0, 20.0, 100.0, 100.0),
    )

    # Object 2: Large near center of chute
    det2 = DetectionResult(
        class_id=20,
        class_name="pet_bottle",
        confidence=0.92,
        bbox=(200.0, 200.0, 440.0, 440.0),
    )

    winner, ranked = prioritizer.prioritize([det1, det2], frame_w, frame_h)
    assert winner is not None
    assert winner.class_name == "pet_bottle"
    assert len(ranked) == 2
    assert ranked[0].class_name == "pet_bottle"


def test_temporal_voting_buffer():
    """Verify 5-frame voting requires consensus threshold before returning stable decision."""
    buffer = TemporalVotingBuffer(window_size=5, min_consensus_ratio=0.60)

    det = DetectionResult(class_id=1, class_name="banana_peel", confidence=0.90, bbox=(10, 10, 50, 50))

    # Push 3 frames (not full yet)
    buffer.add_observation(det)
    buffer.add_observation(det)
    buffer.add_observation(det)
    assert buffer.evaluate_consensus() is None

    # Push 2 more identical frames (now full with 100% consensus)
    buffer.add_observation(det)
    buffer.add_observation(det)

    decision = buffer.evaluate_consensus()
    assert decision is not None
    assert decision.is_stable is True
    assert decision.confirmed_class == "banana_peel"
    assert decision.winning_votes == 5
    assert decision.consensus_ratio == 1.0


def test_material_refiner_fusion():
    """Verify material classification confidence fusion."""
    classifier = MaterialClassifier()
    dummy_frame = np.ones((640, 640, 3), dtype=np.uint8) * 200
    det = DetectionResult(class_id=20, class_name="pet_bottle", confidence=0.80, bbox=(100, 100, 300, 300))

    refined = classifier.refine_detection(dummy_frame, det)
    assert refined.material is not None
    assert 0.0 <= refined.confidence <= 1.0


def test_contamination_detector():
    """Verify contamination detector assesses clean vs dirty textures."""
    detector = ContaminationDetector(threshold=0.60)
    clean_crop = np.ones((100, 100, 3), dtype=np.uint8) * 240
    is_dirty, conf = detector.predict(clean_crop)
    assert is_dirty is False
