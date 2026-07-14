"""
Sliding-window majority voter with consensus-conditioned confidence.

After the active detection window closes, the state machine passes each
track's per-frame predictions to the voter. The voter aggregates them into
one stable decision per track.

Why majority vote instead of simple averaging?
- A single misclassified frame shouldn't flip the decision.
- Confidence averaging across all frames dilutes the signal when some frames
  have poor visibility (hand occluding the item). Consensus-conditioned
  confidence only averages frames that agreed with the winning label.

Why not temporal fusion (LSTM, attention, etc.)?
- The Jetson Orin Nano can't sustain the extra compute, and for the bin
  use case (one item at a time, 1–3 seconds of observation) majority vote
  is empirically sufficient.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from smartbin.config import VoterConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoteResult:
    """Result of the majority vote for a single tracked item."""

    track_id: int
    winning_class: str
    consensus_confidence: float  # Mean confidence of agreeing frames only
    agreeing_frames: int  # Number of frames that predicted the winning class
    total_frames: int  # Total frames this track appeared in
    is_certain: bool  # True if consensus ratio >= threshold


class MajorityVoter:
    """
    Aggregates per-frame class predictions into one final decision per track.

    Algorithm per track:
    1. Count occurrences of each class label.
    2. Winning label = most frequent. Ties broken by highest total confidence.
    3. Consensus-conditioned confidence = mean confidence of ONLY the frames
       that predicted the winning label.
    4. If winning_count / total_frames < min_consensus_ratio → uncertain.
    """

    def __init__(self, config: VoterConfig) -> None:
        self._min_consensus_ratio = config.min_consensus_ratio

    def vote(
        self,
        track_histories: Dict[int, List[Tuple[str, float]]],
        min_frames: int = 1,
    ) -> List[VoteResult]:
        """
        Run majority vote on all tracks.

        Args:
            track_histories: Mapping of track_id → list of (class_name, confidence)
                             tuples, one per frame the track appeared in.
            min_frames: Minimum number of frames a track must have to produce
                        a valid vote. Tracks with fewer frames are dropped.

        Returns:
            List of VoteResult, one per eligible track.
        """
        results: List[VoteResult] = []

        for track_id, predictions in track_histories.items():
            if len(predictions) < min_frames:
                logger.debug(
                    "Voter: track %d dropped (only %d frames, need %d)",
                    track_id,
                    len(predictions),
                    min_frames,
                )
                continue

            result = self._vote_single_track(track_id, predictions)
            results.append(result)

        return results

    def _vote_single_track(
        self,
        track_id: int,
        predictions: List[Tuple[str, float]],
    ) -> VoteResult:
        """Compute majority vote for one track."""
        total_frames = len(predictions)

        # Step 1: Count class occurrences
        class_names = [cls for cls, _ in predictions]
        counts = Counter(class_names)

        # Step 2: Find winning class.
        # Ties broken by highest total confidence for that class.
        winning_class = max(
            counts.keys(),
            key=lambda cls: (
                counts[cls],
                sum(conf for c, conf in predictions if c == cls),
            ),
        )
        agreeing_frames = counts[winning_class]

        # Step 3: Consensus-conditioned confidence — average confidence
        # only from frames that agreed with the winning label.
        agreeing_confs = [
            conf for cls, conf in predictions if cls == winning_class
        ]
        consensus_confidence = (
            sum(agreeing_confs) / len(agreeing_confs) if agreeing_confs else 0.0
        )

        # Step 4: Check consensus ratio
        consensus_ratio = agreeing_frames / total_frames
        is_certain = consensus_ratio >= self._min_consensus_ratio

        logger.debug(
            "Voter: track %d → %s (%.3f conf, %d/%d frames, %s)",
            track_id,
            winning_class,
            consensus_confidence,
            agreeing_frames,
            total_frames,
            "CERTAIN" if is_certain else "UNCERTAIN",
        )

        return VoteResult(
            track_id=track_id,
            winning_class=winning_class,
            consensus_confidence=consensus_confidence,
            agreeing_frames=agreeing_frames,
            total_frames=total_frames,
            is_certain=is_certain,
        )
