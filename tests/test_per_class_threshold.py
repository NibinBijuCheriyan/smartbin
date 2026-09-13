"""Tests for per-class confidence threshold functionality.

Covers:
- ModelConfig.get_confidence_threshold() with scalar, dict, and unknown class
- ModelConfig.get_min_confidence_threshold()
- MajorityVoter per-class threshold marking uncertain
"""

from __future__ import annotations

import pytest

from smartbin.config import ModelConfig, VoterConfig
from smartbin.voter import MajorityVoter


class TestModelConfigThreshold:
    """Tests for ModelConfig.get_confidence_threshold."""

    def test_scalar_threshold_returns_same_for_all_classes(self):
        cfg = ModelConfig(confidence_threshold=0.40)
        assert cfg.get_confidence_threshold("plastic") == 0.40
        assert cfg.get_confidence_threshold("other") == 0.40
        assert cfg.get_confidence_threshold("unknown") == 0.40

    def test_dict_threshold_returns_per_class_value(self):
        cfg = ModelConfig(confidence_threshold={
            "plastic": 0.40, "paper": 0.40, "metal": 0.40,
            "glass": 0.40, "other": 0.55,
        })
        assert cfg.get_confidence_threshold("plastic") == 0.40
        assert cfg.get_confidence_threshold("other") == 0.55

    def test_dict_threshold_case_insensitive(self):
        cfg = ModelConfig(confidence_threshold={
            "plastic": 0.40, "other": 0.55,
        })
        assert cfg.get_confidence_threshold("Plastic") == 0.40
        assert cfg.get_confidence_threshold("OTHER") == pytest.approx(0.55)

    def test_dict_threshold_unknown_class_falls_back_to_min(self):
        cfg = ModelConfig(confidence_threshold={
            "plastic": 0.40, "other": 0.55,
        })
        # Unknown class should get min(0.40, 0.55) = 0.40
        assert cfg.get_confidence_threshold("unknown_class") == 0.40

    def test_get_min_confidence_threshold_scalar(self):
        cfg = ModelConfig(confidence_threshold=0.35)
        assert cfg.get_min_confidence_threshold() == 0.35

    def test_get_min_confidence_threshold_dict(self):
        cfg = ModelConfig(confidence_threshold={
            "plastic": 0.40, "paper": 0.40, "metal": 0.40,
            "glass": 0.40, "other": 0.55,
        })
        assert cfg.get_min_confidence_threshold() == 0.40


class TestVoterPerClassThreshold:
    """Tests for MajorityVoter with per-class confidence thresholds."""

    def test_other_class_below_threshold_marked_uncertain(self):
        """'other' at 0.50 consensus conf should be uncertain with 0.55 threshold."""
        voter = MajorityVoter(
            VoterConfig(min_consensus_ratio=0.4),
            confidence_thresholds={"plastic": 0.40, "other": 0.55},
        )
        histories = {
            1: [("other", 0.50), ("other", 0.48), ("other", 0.52)],
        }
        results = voter.vote(histories)
        assert len(results) == 1
        r = results[0]
        assert r.winning_class == "other"
        assert r.is_certain is False  # 0.50 avg < 0.55 threshold

    def test_plastic_above_threshold_marked_certain(self):
        """'plastic' at 0.50 consensus conf should be certain with 0.40 threshold."""
        voter = MajorityVoter(
            VoterConfig(min_consensus_ratio=0.4),
            confidence_thresholds={"plastic": 0.40, "other": 0.55},
        )
        histories = {
            1: [("plastic", 0.50), ("plastic", 0.48), ("plastic", 0.52)],
        }
        results = voter.vote(histories)
        assert len(results) == 1
        r = results[0]
        assert r.winning_class == "plastic"
        assert r.is_certain is True  # 0.50 avg >= 0.40 threshold

    def test_no_thresholds_backward_compatible(self):
        """Voter without confidence_thresholds behaves as before."""
        voter = MajorityVoter(VoterConfig(min_consensus_ratio=0.4))
        histories = {
            1: [("other", 0.50), ("other", 0.48), ("other", 0.52)],
        }
        results = voter.vote(histories)
        r = results[0]
        assert r.winning_class == "other"
        assert r.is_certain is True  # 3/3 = 1.0 >= 0.4 consensus ratio, no threshold check

    def test_scalar_threshold_applies_to_all_classes(self):
        """Scalar confidence_threshold applies uniformly."""
        voter = MajorityVoter(
            VoterConfig(min_consensus_ratio=0.4),
            confidence_thresholds=0.60,
        )
        histories = {
            1: [("plastic", 0.50), ("plastic", 0.55), ("plastic", 0.52)],
        }
        results = voter.vote(histories)
        r = results[0]
        assert r.winning_class == "plastic"
        assert r.is_certain is False  # ~0.523 avg < 0.60 threshold
