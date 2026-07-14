"""
Decision event dataclass and output hooks.

The DecisionEvent is the structured output of the pipeline — it represents
one finalized waste-classification decision for a single tracked item.

Hooks consume these events and route them to storage, logging, or downstream
systems (bin actuation, cloud dashboards, etc.).
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionEvent:
    """
    A finalized classification decision for one tracked waste item.

    Produced by the majority voter after the active detection window closes.
    Downstream systems (bin actuation, reward calculation) consume this.
    """

    track_id: int
    item_class: str
    confidence: float  # Consensus-conditioned: mean conf of agreeing frames only
    frame_count: int  # Frames where the winning class was detected
    total_frames: int  # Total frames this track appeared in
    is_certain: bool  # True if consensus ratio met threshold
    timestamp: str  # ISO 8601 UTC timestamp

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to a compact JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @staticmethod
    def create(
        track_id: int,
        item_class: str,
        confidence: float,
        frame_count: int,
        total_frames: int,
        is_certain: bool,
    ) -> DecisionEvent:
        """Factory with automatic UTC timestamp."""
        return DecisionEvent(
            track_id=track_id,
            item_class=item_class,
            confidence=round(confidence, 4),
            frame_count=frame_count,
            total_frames=total_frames,
            is_certain=is_certain,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# ---------------------------------------------------------------------------
# Hook interface
# ---------------------------------------------------------------------------


class DecisionHook(ABC):
    """
    Abstract base class for decision output hooks.

    Implement `on_decision` to route finalized classification events to
    any downstream system — file logging, MQTT, cloud API, etc.
    """

    @abstractmethod
    def on_decision(self, event: DecisionEvent) -> None:
        """Called once per finalized decision event."""

    def on_batch(self, events: List[DecisionEvent]) -> None:
        """Called with all decisions from one active window. Default: iterate."""
        for event in events:
            self.on_decision(event)

    def close(self) -> None:
        """Clean up resources (file handles, connections). Default: no-op."""


# ---------------------------------------------------------------------------
# Built-in hooks
# ---------------------------------------------------------------------------


class LoggingHook(DecisionHook):
    """Emits each decision via Python's logging at INFO level."""

    def on_decision(self, event: DecisionEvent) -> None:
        certainty = "CERTAIN" if event.is_certain else "UNCERTAIN"
        logger.info(
            "[DECISION] track=%d class=%s conf=%.3f frames=%d/%d %s",
            event.track_id,
            event.item_class,
            event.confidence,
            event.frame_count,
            event.total_frames,
            certainty,
        )


class JsonlFileHook(DecisionHook):
    """
    Appends each decision as a JSON line to a .jsonl file.

    This format is trivially parseable, appendable, and works well for
    later batch upload to cloud analytics or a dashboard.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")
        logger.info("Decision JSONL log: %s", os.path.abspath(path))

    def on_decision(self, event: DecisionEvent) -> None:
        self._file.write(event.to_json() + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()


# ---------------------------------------------------------------------------
# Future hook stubs — extension points for downstream integration
# ---------------------------------------------------------------------------


# TODO: MqttHook — publish DecisionEvent to an MQTT topic for bin actuation.
#   class MqttHook(DecisionHook):
#       def __init__(self, broker: str, topic: str): ...
#       def on_decision(self, event): client.publish(topic, event.to_json())

# TODO: CloudSyncHook — batch-upload decisions to a cloud dashboard/API.
#   class CloudSyncHook(DecisionHook):
#       def __init__(self, endpoint: str, api_key: str): ...
#       def on_batch(self, events): requests.post(endpoint, json=[...])
