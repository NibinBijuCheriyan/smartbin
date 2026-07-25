"""
Trigger gate — low-compute motion/presence detection.

Keeps the YOLO detector dormant until an item or hand actually approaches
the bin. This is critical for power efficiency on edge devices: the detector
should NOT run continuously when nothing is happening.

Design choice: frame differencing (background subtraction) is used because
it's extremely cheap (~1ms per frame on any hardware) and sufficient for
detecting "something moved in front of the bin". More sophisticated triggers
(IR sensor, weight sensor) can be added by subclassing BaseTrigger.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

from smartbin.config import TriggerConfig

logger = logging.getLogger(__name__)


class BaseTrigger(ABC):
    """
    Abstract trigger interface.

    Subclass this for hardware-based triggers (IR, weight sensor).
    The state machine only calls `check()` — it doesn't care how
    the trigger decides.
    """

    @abstractmethod
    def check(self, frame: np.ndarray) -> bool:
        """Return True if the trigger condition is met (item approaching)."""

    def reset(self) -> None:
        """Reset internal state. Called when the state machine returns to IDLE."""


class FrameDiffTrigger(BaseTrigger):
    """
    Motion detection via frame differencing (background subtraction).

    Maintains a reference background frame and compares each new frame
    against it. If enough pixels have changed, the trigger fires.

    The background adapts slowly via exponential moving average while the
    system is IDLE, so gradual lighting changes don't cause false triggers.
    """

    def __init__(self, config: TriggerConfig) -> None:
        self._threshold = config.motion_threshold
        self._area_fraction = config.area_fraction
        self._alpha = config.background_alpha
        self._roi = self._parse_roi(config.roi)
        self._background: Optional[np.ndarray] = None  # Grayscale float32

    @staticmethod
    def _parse_roi(
        roi: Optional[list],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Validate and convert ROI list to (x1, y1, x2, y2) tuple."""
        if roi is None:
            return None
        if len(roi) != 4:
            raise ValueError(f"ROI must be [x1, y1, x2, y2], got {roi}")
        return tuple(roi)  # type: ignore[return-value]

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract the region of interest from a frame."""
        if self._roi is None:
            return frame
        x1, y1, x2, y2 = self._roi
        return frame[y1:y2, x1:x2]

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale and blur to reduce noise."""
        roi = self._crop_roi(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        return blurred.astype(np.float32)

    def check(self, frame: np.ndarray) -> bool:
        """
        Check whether motion is detected in the current frame.

        Returns True if the fraction of changed pixels exceeds the
        configured area_fraction threshold.
        """
        current = self._preprocess(frame)

        # First frame — initialise background, no trigger
        if self._background is None:
            self._background = current.copy()
            logger.debug("Trigger: background initialized")
            return False

        # Compute absolute difference
        diff = np.abs(current - self._background)
        changed_mask = diff > self._threshold
        changed_fraction = np.count_nonzero(changed_mask) / changed_mask.size

        triggered = bool(changed_fraction > self._area_fraction)

        if triggered:
            logger.debug(
                "Trigger: motion detected (%.4f > %.4f)",
                changed_fraction,
                self._area_fraction,
            )
        else:
            # Slowly adapt background while idle (EMA)
            cv2.accumulateWeighted(current, self._background, self._alpha)

        return triggered

    def reset(self) -> None:
        """
        Reset the background reference.

        Called when the state machine returns to IDLE so the next
        idle period starts with a fresh background.
        """
        self._background = None
        logger.debug("Trigger: background reset")


class HandPresenceTrigger(BaseTrigger):
    """
    Trigger based on hand presence.

    Uses HandTracker to detect if a hand enters the frame or specified ROI.
    """

    def __init__(self, config: TriggerConfig) -> None:
        from smartbin.hand_tracker import HandTracker
        self._tracker = HandTracker()

    def check(self, frame: np.ndarray) -> bool:
        hands = self._tracker.detect_and_track(frame)
        return len(hands) > 0

    def reset(self) -> None:
        self._tracker.reset()


def create_trigger(config: TriggerConfig) -> BaseTrigger:
    """Factory function — returns the appropriate trigger for the config."""
    if config.method == "frame_diff":
        return FrameDiffTrigger(config)
    elif config.method == "hand_presence":
        return HandPresenceTrigger(config)
    raise ValueError(f"Unknown trigger method: {config.method}")

