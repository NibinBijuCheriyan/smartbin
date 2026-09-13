"""Track-scoped temporal voting for safe physical actuation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from smartbin_v2.inference.engine import DetectionResult
from smartbin_v2.utils.logger import get_logger

logger = get_logger("temporal_voter")


@dataclass
class VotingDecision:
    confirmed_class: str
    class_id: int
    mean_confidence: float
    consensus_ratio: float
    is_stable: bool
    track_id: int
    reference_detection: DetectionResult
    total_votes: int
    winning_votes: int


class TemporalVotingBuffer:
    """Accepts only consecutive observations of one track for a decision."""

    def __init__(self, window_size: int = 5, min_consensus_ratio: float = 0.60) -> None:
        self.window_size = window_size
        self.min_consensus_ratio = min_consensus_ratio
        self._history: List[DetectionResult] = []
        self._track_id: Optional[int] = None

    def add_observation(self, detection: Optional[DetectionResult]) -> None:
        if detection is None:
            self.reset()
            return
        if self._track_id is not None and detection.track_id != self._track_id:
            self.reset()
        self._track_id = detection.track_id
        self._history.append(detection)
        if len(self._history) > self.window_size:
            self._history.pop(0)

    def reset(self) -> None:
        self._history.clear()
        self._track_id = None

    def evaluate_consensus(self) -> Optional[VotingDecision]:
        if len(self._history) < self.window_size or self._track_id is None:
            return None
        if any(detection.track_id != self._track_id for detection in self._history):
            return None

        votes: Counter[str] = Counter(detection.class_name for detection in self._history)
        winning_class, winning_votes = votes.most_common(1)[0]
        consensus_ratio = winning_votes / self.window_size
        confidences: Dict[str, List[float]] = defaultdict(list)
        for detection in self._history:
            confidences[detection.class_name].append(detection.confidence)
        mean_confidence = sum(confidences[winning_class]) / winning_votes
        reference = next(d for d in reversed(self._history) if d.class_name == winning_class)

        return VotingDecision(
            confirmed_class=winning_class,
            class_id=reference.class_id,
            mean_confidence=round(mean_confidence, 3),
            consensus_ratio=round(consensus_ratio, 3),
            is_stable=consensus_ratio >= self.min_consensus_ratio,
            track_id=self._track_id,
            reference_detection=reference,
            total_votes=self.window_size,
            winning_votes=winning_votes,
        )