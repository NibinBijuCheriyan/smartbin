"""
State machine and frame buffer for the Smartbin pipeline.

Manages the IDLE → ACTIVE → IDLE lifecycle:
- IDLE: detector is dormant, only the trigger gate runs (cheap).
- ACTIVE: detector runs on every frame, results are buffered.
- Finalization: when the active window ends, buffered detections are
  aggregated by track ID and passed to the majority voter.

The state machine is the central coordinator — it decides when to start
detecting, when to stop, and when to produce final decisions.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from smartbin.config import BufferConfig, VoterConfig
from smartbin.decision import DecisionEvent
from smartbin.detector import Detection
from smartbin.voter import MajorityVoter

logger = logging.getLogger(__name__)


class BinState(Enum):
    """Pipeline operating state."""

    IDLE = auto()  # Detector dormant, trigger gate running
    ACTIVE = auto()  # Detector running, buffering detections


class StateMachine:
    """
    Manages the IDLE ↔ ACTIVE lifecycle with a rolling frame buffer.

    Transition rules:
    - IDLE + trigger fires          → ACTIVE (reset buffer, start detecting)
    - ACTIVE + detections present   → buffer detections, reset idle counter
    - ACTIVE + no detections        → increment idle counter
    - ACTIVE + buffer full          → finalize → IDLE
    - ACTIVE + idle counter exceeds threshold → finalize → IDLE
    """

    def __init__(
        self,
        buffer_config: BufferConfig,
        voter_config: VoterConfig,
    ) -> None:
        self._window_size = buffer_config.active_window_size
        self._idle_timeout = buffer_config.idle_timeout_frames
        self._min_frames = buffer_config.min_frames_for_decision
        self._voter = MajorityVoter(voter_config)

        # Runtime state
        self._state = BinState.IDLE
        self._frame_buffer: deque = deque(maxlen=self._window_size)
        self._idle_counter = 0
        self._frame_count = 0  # Frames processed in the current active window

    # -- Properties ----------------------------------------------------------

    @property
    def state(self) -> BinState:
        """Current pipeline state."""
        return self._state

    @property
    def frame_count(self) -> int:
        """Number of frames processed in the current active window."""
        return self._frame_count

    @property
    def buffer_size(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._frame_buffer)

    # -- State transitions ---------------------------------------------------

    def activate(self) -> None:
        """
        Transition from IDLE to ACTIVE.

        Called by the pipeline when the trigger gate fires.
        Resets all buffers and counters for a fresh detection window.
        """
        if self._state == BinState.ACTIVE:
            logger.warning("activate() called while already ACTIVE — ignored")
            return

        self._state = BinState.ACTIVE
        self._frame_buffer.clear()
        self._idle_counter = 0
        self._frame_count = 0
        logger.info("State: IDLE → ACTIVE")

    def feed(
        self, detections: List[Detection]
    ) -> Optional[List[DecisionEvent]]:
        """
        Feed one frame's detections into the buffer.

        This is the main per-frame call while ACTIVE. Returns None if the
        window is still open, or a list of DecisionEvents if finalization
        was triggered.

        Args:
            detections: List of Detection objects from the current frame.
                        May be empty (no objects detected).

        Returns:
            None if still buffering, or List[DecisionEvent] on finalization.
        """
        if self._state != BinState.ACTIVE:
            logger.warning("feed() called while IDLE — ignored")
            return None

        self._frame_count += 1

        # Only buffer detections that have a valid track ID
        tracked = [d for d in detections if d.track_id >= 0]
        self._frame_buffer.append(tracked)

        if tracked:
            self._idle_counter = 0
        else:
            self._idle_counter += 1
            logger.debug(
                "No tracked detections (%d/%d idle frames)",
                self._idle_counter,
                self._idle_timeout,
            )

        # Check finalization conditions
        should_finalize = False

        if self._frame_count >= self._window_size:
            logger.info("Active window full (%d frames) — finalizing", self._frame_count)
            should_finalize = True

        elif self._idle_counter >= self._idle_timeout:
            logger.info(
                "Idle timeout (%d consecutive empty frames) — finalizing",
                self._idle_counter,
            )
            should_finalize = True

        if should_finalize:
            return self._finalize()

        return None

    def _finalize(self) -> List[DecisionEvent]:
        """
        Finalize the active window: aggregate detections by track ID,
        run majority vote, and produce DecisionEvents.

        Transitions the state machine back to IDLE.
        """
        # Build track histories: track_id → [(class_name, confidence, hand_id, is_held_by_hand), ...]
        track_histories: Dict[int, List[Tuple]] = defaultdict(list)

        for frame_detections in self._frame_buffer:
            for det in frame_detections:
                track_histories[det.track_id].append(
                    (det.class_name, det.confidence, det.hand_id, det.is_held_by_hand)
                )

        logger.info(
            "Finalizing: %d frames, %d unique tracks",
            len(self._frame_buffer),
            len(track_histories),
        )

        # Run majority vote
        vote_results = self._voter.vote(
            track_histories, min_frames=self._min_frames
        )

        # Convert to DecisionEvents
        events = [
            DecisionEvent.create(
                track_id=vr.track_id,
                item_class=vr.winning_class,
                confidence=vr.consensus_confidence,
                frame_count=vr.agreeing_frames,
                total_frames=vr.total_frames,
                is_certain=vr.is_certain,
                hand_id=vr.hand_id,
                is_held_by_hand=vr.is_held_by_hand,
            )
            for vr in vote_results
        ]

        # Transition back to IDLE
        self._state = BinState.IDLE
        self._frame_buffer.clear()
        self._idle_counter = 0
        self._frame_count = 0
        logger.info("State: ACTIVE → IDLE (%d decisions emitted)", len(events))

        return events

    def force_finalize(self) -> Optional[List[DecisionEvent]]:
        """
        Force finalization regardless of window state.

        Useful for shutdown/cleanup — ensures any buffered detections
        are processed before the pipeline stops.
        """
        if self._state != BinState.ACTIVE:
            return None
        if len(self._frame_buffer) == 0:
            self._state = BinState.IDLE
            return []
        return self._finalize()
