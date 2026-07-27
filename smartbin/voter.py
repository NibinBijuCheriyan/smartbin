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
from typing import Dict, List, Optional, Tuple

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
    hand_id: Optional[int] = None
    is_held_by_hand: bool = False
    raw_yolo_class: Optional[str] = None
    refiner_class: Optional[str] = None
    is_refined: bool = False


class MajorityVoter:
    """
    Aggregates per-frame class predictions into one final decision per track.

    Algorithm per track:
    1. Count occurrences of each class label.
    2. Winning label = most frequent. Ties broken by highest total confidence.
    3. Consensus-conditioned confidence = mean confidence of ONLY the frames
       that predicted the winning label.
    4. Aggregate hand association IDs.
    5. If winning_count / total_frames < min_consensus_ratio → uncertain.
    """

    def __init__(self, config: VoterConfig) -> None:
        self._min_consensus_ratio = config.min_consensus_ratio

    def vote(
        self,
        track_histories: Dict[int, List[Tuple]],
        min_frames: int = 1,
    ) -> List[VoteResult]:
        """
        Run majority vote on all tracks.

        Args:
            track_histories: Mapping of track_id → list of tuples:
                             (class_name, confidence) or (class_name, confidence, hand_id, is_held_by_hand, raw_yolo_class, is_refined)
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
        predictions: List[Tuple],
    ) -> VoteResult:
        """Compute majority vote for one track."""
        total_frames = len(predictions)

        # Handle predictions tuples of varying length (2, 4, or 6 items)
        class_names = []
        hand_ids = []
        held_flags = []
        raw_yolo_classes = []
        is_refined_flags = []

        for item in predictions:
            class_names.append(item[0])
            if len(item) >= 4:
                if item[2] is not None:
                    hand_ids.append(item[2])
                held_flags.append(item[3])
            if len(item) >= 6:
                if item[4] is not None:
                    raw_yolo_classes.append(item[4])
                is_refined_flags.append(item[5])

        counts = Counter(class_names)

        # Step 2: Find winning class.
        winning_class = max(
            counts.keys(),
            key=lambda cls: (
                counts[cls],
                sum(item[1] for item in predictions if item[0] == cls),
            ),
        )
        agreeing_frames = counts[winning_class]

        # Step 3: Consensus-conditioned confidence
        agreeing_confs = [
            item[1] for item in predictions if item[0] == winning_class
        ]
        consensus_confidence = (
            sum(agreeing_confs) / len(agreeing_confs) if agreeing_confs else 0.0
        )

        # Step 4: Check consensus ratio
        consensus_ratio = agreeing_frames / total_frames
        is_certain = consensus_ratio >= self._min_consensus_ratio

        # Determine hand tracking summary
        winning_hand_id = Counter(hand_ids).most_common(1)[0][0] if hand_ids else None
        is_held_by_hand = (sum(held_flags) > total_frames / 2.0) if held_flags else False

        # Determine raw YOLO class and refinement status
        most_common_raw_yolo = Counter(raw_yolo_classes).most_common(1)[0][0] if raw_yolo_classes else None
        is_refined = any(is_refined_flags) if is_refined_flags else False
        refiner_class = winning_class if is_refined else None

        logger.debug(
            "Voter: track %d → %s (%.3f conf, %d/%d frames, %s, hand=%s, refined=%s)",
            track_id,
            winning_class,
            consensus_confidence,
            agreeing_frames,
            total_frames,
            "CERTAIN" if is_certain else "UNCERTAIN",
            str(winning_hand_id),
            str(is_refined),
        )

        return VoteResult(
            track_id=track_id,
            winning_class=winning_class,
            consensus_confidence=consensus_confidence,
            agreeing_frames=agreeing_frames,
            total_frames=total_frames,
            is_certain=is_certain,
            hand_id=winning_hand_id,
            is_held_by_hand=is_held_by_hand,
            raw_yolo_class=most_common_raw_yolo,
            refiner_class=refiner_class,
            is_refined=is_refined,
        )
