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
from typing import List, Optional

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
    hand_id: Optional[int] = None
    is_held_by_hand: bool = False

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
        hand_id: Optional[int] = None,
        is_held_by_hand: bool = False,
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
            hand_id=hand_id,
            is_held_by_hand=is_held_by_hand,
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
# HTTP Webhook hook — real actuation endpoint
# ---------------------------------------------------------------------------


class WebhookHook(DecisionHook):
    """
    POST each DecisionEvent as JSON to an HTTP webhook endpoint.

    This closes the loop from "classified item" to "physical or system action"
    — the webhook can trigger a servo/flap controller, update a dashboard, or
    feed into a rewards system.

    **Hook Contract:**

    - Each decision is POSTed individually as JSON to the configured URL.
    - On transient failures (connection error, timeout, HTTP 5xx), the hook
      retries up to `max_retries` times with exponential backoff (1s, 2s, 4s).
    - After exhausting retries, the event is **dropped** and an error is logged.
      Events are NOT queued for re-delivery after pipeline restart.
    - The hook runs delivery in a background thread to avoid blocking the
      frame processing loop. A bounded queue prevents memory exhaustion.
    - The hook is **best-effort**: the pipeline continues processing
      regardless of webhook success or failure.
    - Duplicate delivery is possible if the remote server accepts the POST
      but the response is lost (at-least-once semantics when retries fire).

    Args:
        url: HTTP(S) endpoint to POST decisions to.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on transient failures.
        max_queue_size: Maximum pending events in the delivery queue.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 5.0,
        max_retries: int = 3,
        max_queue_size: int = 100,
    ) -> None:
        import queue
        import threading

        self._url = url
        self._timeout = timeout
        self._max_retries = max_retries

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown = threading.Event()
        self._worker = threading.Thread(
            target=self._delivery_loop, daemon=True, name="webhook-hook"
        )
        self._worker.start()
        logger.info("WebhookHook initialized: %s (timeout=%.1fs, retries=%d)",
                     url, timeout, max_retries)

    def on_decision(self, event: DecisionEvent) -> None:
        """Queue a decision for async delivery. Drops if queue is full."""
        try:
            self._queue.put_nowait(event)
        except Exception:
            logger.warning(
                "WebhookHook queue full — dropping decision for track %d",
                event.track_id,
            )

    def _delivery_loop(self) -> None:
        """Background thread: drain queue and POST events."""
        import time

        while not self._shutdown.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except Exception:
                continue

            self._deliver_with_retry(event)
            self._queue.task_done()

    def _deliver_with_retry(self, event: DecisionEvent) -> None:
        """POST a single event with exponential backoff retry."""
        import time
        import urllib.request
        import urllib.error

        payload = event.to_json().encode("utf-8")
        headers = {"Content-Type": "application/json"}

        for attempt in range(self._max_retries + 1):
            try:
                req = urllib.request.Request(
                    self._url,
                    data=payload,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    status = resp.getcode()
                    if 200 <= status < 300:
                        logger.debug(
                            "WebhookHook delivered track=%d (%d)",
                            event.track_id, status,
                        )
                        return
                    elif status >= 500:
                        # Server error — retry
                        raise urllib.error.HTTPError(
                            self._url, status, "Server error", {}, None
                        )
                    else:
                        # Client error (4xx) — don't retry
                        logger.error(
                            "WebhookHook: HTTP %d for track=%d — not retrying",
                            status, event.track_id,
                        )
                        return

            except (urllib.error.URLError, OSError, urllib.error.HTTPError) as e:
                if attempt < self._max_retries:
                    backoff = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        "WebhookHook: attempt %d/%d failed for track=%d: %s "
                        "(retrying in %ds)",
                        attempt + 1, self._max_retries + 1,
                        event.track_id, e, backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        "WebhookHook: all %d attempts failed for track=%d: %s "
                        "— event dropped",
                        self._max_retries + 1, event.track_id, e,
                    )

    def close(self) -> None:
        """Shutdown the delivery thread and drain remaining events."""
        self._shutdown.set()
        if self._worker.is_alive():
            self._worker.join(timeout=10)
        logger.info("WebhookHook shut down.")


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

