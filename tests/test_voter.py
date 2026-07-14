"""
Tests for the majority voter module.

Exercises the consensus-conditioned confidence calculation, tie-breaking,
minimum frame requirements, and uncertainty marking — all with synthetic
prediction data (no model or camera required).
"""

from __future__ import annotations

import pytest

from smartbin.config import VoterConfig
from smartbin.voter import MajorityVoter


class TestMajorityVoter:
    """Tests for MajorityVoter."""

    def _voter(self, min_consensus_ratio: float = 0.4) -> MajorityVoter:
        return MajorityVoter(VoterConfig(min_consensus_ratio=min_consensus_ratio))

    # -- Unanimous vote ---------------------------------------------------

    def test_unanimous_vote(self):
        """All frames agree on the same class."""
        voter = self._voter()
        histories = {
            1: [("plastic", 0.9), ("plastic", 0.85), ("plastic", 0.92)],
        }
        results = voter.vote(histories)
        assert len(results) == 1

        r = results[0]
        assert r.track_id == 1
        assert r.winning_class == "plastic"
        assert r.agreeing_frames == 3
        assert r.total_frames == 3
        assert r.is_certain is True
        # Consensus confidence = mean of all 3 (all agree)
        expected_conf = (0.9 + 0.85 + 0.92) / 3
        assert abs(r.consensus_confidence - expected_conf) < 1e-6

    # -- Majority with noise ----------------------------------------------

    def test_majority_with_noise(self):
        """Most frames agree, a few disagree. Confidence excludes noisy frames."""
        voter = self._voter()
        histories = {
            2: [
                ("paper", 0.8),
                ("paper", 0.75),
                ("metal", 0.6),  # noise
                ("paper", 0.85),
                ("plastic", 0.4),  # noise
            ],
        }
        results = voter.vote(histories)
        r = results[0]

        assert r.winning_class == "paper"
        assert r.agreeing_frames == 3
        assert r.total_frames == 5
        # Consensus confidence = mean of the 3 "paper" confidences only
        expected_conf = (0.8 + 0.75 + 0.85) / 3
        assert abs(r.consensus_confidence - expected_conf) < 1e-6
        assert r.is_certain is True  # 3/5 = 0.6 ≥ 0.4

    # -- Tie-breaking by total confidence ---------------------------------

    def test_tie_broken_by_confidence(self):
        """When two classes have equal count, the one with higher total confidence wins."""
        voter = self._voter()
        histories = {
            3: [
                ("glass", 0.7),
                ("glass", 0.65),
                ("metal", 0.9),
                ("metal", 0.95),
            ],
        }
        results = voter.vote(histories)
        r = results[0]

        # glass total = 1.35, metal total = 1.85 → metal wins
        assert r.winning_class == "metal"
        assert r.agreeing_frames == 2

    # -- Consensus ratio threshold ----------------------------------------

    def test_uncertain_when_low_consensus(self):
        """Decision marked uncertain when consensus ratio is below threshold."""
        voter = self._voter(min_consensus_ratio=0.6)
        histories = {
            4: [
                ("plastic", 0.5),
                ("paper", 0.6),
                ("metal", 0.7),
                ("glass", 0.4),
                ("plastic", 0.55),
            ],
        }
        results = voter.vote(histories)
        r = results[0]

        # "plastic" wins with 2/5 = 0.4, which is < 0.6 threshold
        assert r.winning_class == "plastic"
        assert r.is_certain is False

    def test_certain_at_threshold_boundary(self):
        """Decision is certain when consensus ratio exactly equals threshold."""
        voter = self._voter(min_consensus_ratio=0.5)
        histories = {
            5: [
                ("ewaste", 0.8),
                ("ewaste", 0.75),
                ("organic", 0.6),
                ("organic", 0.55),
            ],
        }
        results = voter.vote(histories)
        r = results[0]

        # "ewaste" wins by confidence tie-break: 1.55 > 1.15
        # 2/4 = 0.5 = threshold → certain
        assert r.is_certain is True

    # -- Minimum frame requirement ----------------------------------------

    def test_min_frames_filter(self):
        """Tracks with fewer than min_frames are dropped."""
        voter = self._voter()
        histories = {
            6: [("plastic", 0.9)],  # Only 1 frame
            7: [("paper", 0.8), ("paper", 0.85), ("paper", 0.9)],  # 3 frames
        }
        results = voter.vote(histories, min_frames=3)

        assert len(results) == 1
        assert results[0].track_id == 7

    # -- Edge cases -------------------------------------------------------

    def test_empty_input(self):
        """No tracks → no results."""
        voter = self._voter()
        results = voter.vote({})
        assert results == []

    def test_multiple_tracks(self):
        """Multiple tracks are voted independently."""
        voter = self._voter()
        histories = {
            10: [("plastic", 0.9), ("plastic", 0.85)],
            11: [("metal", 0.8), ("metal", 0.75)],
        }
        results = voter.vote(histories)
        assert len(results) == 2

        result_map = {r.track_id: r for r in results}
        assert result_map[10].winning_class == "plastic"
        assert result_map[11].winning_class == "metal"

    def test_single_frame_track(self):
        """A track with exactly 1 frame still produces a valid result."""
        voter = self._voter()
        histories = {
            12: [("organic", 0.65)],
        }
        results = voter.vote(histories, min_frames=1)
        r = results[0]

        assert r.winning_class == "organic"
        assert r.agreeing_frames == 1
        assert r.total_frames == 1
        assert abs(r.consensus_confidence - 0.65) < 1e-6
