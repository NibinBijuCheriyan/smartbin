"""
Pipeline orchestrator — thin wiring layer.

Connects the trigger, state machine, detector, voter, and decision hooks
into a single processing loop. Contains no business logic itself — just
the frame-by-frame coordination.

This is the "main loop" of the Smartbin system. It:
1. Opens the video source (webcam or file).
2. Reads frames at a configurable FPS.
3. Routes each frame through the appropriate stage based on state.
4. Handles errors gracefully (camera disconnects, model failures).
"""

from __future__ import annotations

import logging
import platform
import time
from typing import List, Optional

import cv2
import numpy as np

from smartbin.config import SmartbinConfig
from smartbin.decision import (
    DecisionEvent,
    DecisionHook,
    JsonlFileHook,
    LoggingHook,
)
from smartbin.detector import BaseDetector, create_detector
from smartbin.state_machine import BinState, StateMachine
from smartbin.trigger import BaseTrigger, create_trigger

logger = logging.getLogger(__name__)


class SmartbinPipeline:
    """
    Main pipeline orchestrator for the Smartbin waste detection system.

    Wires together: trigger → state machine → detector → voter → hooks.

    Usage:
        config = load_config()
        pipeline = SmartbinPipeline(config)
        pipeline.run()  # Blocks until shutdown
    """

    def __init__(self, config: SmartbinConfig) -> None:
        self._config = config

        # Instantiate components
        self._trigger: BaseTrigger = create_trigger(config.trigger)
        self._state_machine = StateMachine(config.buffer, config.voter)
        self._detector: BaseDetector = create_detector(
            config.model, config.tracker
        )

        # Hand tracking component
        self._hand_tracker = None
        if config.hand_tracking.enabled:
            from smartbin.hand_tracker import HandTracker
            self._hand_tracker = HandTracker(
                confidence_threshold=config.hand_tracking.confidence_threshold,
                max_distance_px=config.hand_tracking.max_hand_distance_px,
            )

        self._current_hands = []
        self._current_detections = []

        # Decision output hooks
        self._hooks: List[DecisionHook] = self._create_hooks(config)

        # FPS limiting
        self._min_frame_interval = (
            1.0 / config.camera.fps_limit if config.camera.fps_limit > 0 else 0
        )

        # Display
        self._show = config.display.show

    def _create_hooks(self, config: SmartbinConfig) -> List[DecisionHook]:
        """Instantiate all configured decision output hooks."""
        hooks: List[DecisionHook] = [LoggingHook()]

        if config.logging.decision_log:
            hooks.append(JsonlFileHook(config.logging.decision_log))

        return hooks

    def run(self) -> None:
        """
        Main processing loop. Blocks until the video source ends or
        KeyboardInterrupt is received.
        """
        source = self._config.camera.source
        logger.info("Opening video source: %s", source)

        cap = self._open_video_source(source)
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open video source: {source}. "
                f"Check camera connection or file path."
            )

        try:
            self._process_loop(cap)
        except KeyboardInterrupt:
            logger.info("Shutdown requested (KeyboardInterrupt)")
        finally:
            self._shutdown(cap)

    def _open_video_source(self, source) -> cv2.VideoCapture:
        """
        Open a video source with platform-appropriate backend.

        On Windows, the default MSMF backend frequently fails for webcams
        (error -1072875772). We try DirectShow first for integer sources.
        """
        # Webcam source on Windows → prefer DirectShow over MSMF
        if isinstance(source, int) and platform.system() == "Windows":
            logger.info("Windows detected — trying DirectShow backend")
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            logger.warning("DirectShow failed, falling back to default backend")

        return cv2.VideoCapture(source)

    def _process_loop(self, cap: cv2.VideoCapture) -> None:
        """Core frame-by-frame processing loop."""
        frame_num = 0
        consecutive_failures = 0
        max_failures = 30  # Give up after this many consecutive read failures

        while True:
            loop_start = time.monotonic()

            # Read frame
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.warning(
                        "Video source ended or %d consecutive read failures — stopping",
                        max_failures,
                    )
                    break
                # Brief pause before retry (camera might reconnect)
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            frame_num += 1

            # Route frame based on current state
            events = self._process_frame(frame, frame_num)

            # Fire hooks for any finalized decisions
            if events:
                self._fire_hooks(events)

            # Display (optional)
            if self._show:
                try:
                    self._display_frame(frame, frame_num)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Display window closed by user")
                        break
                except cv2.error:
                    logger.warning(
                        "GUI display unavailable — disabling --show"
                    )
                    self._show = False

            # FPS limiting
            elapsed = time.monotonic() - loop_start
            if elapsed < self._min_frame_interval:
                time.sleep(self._min_frame_interval - elapsed)

    def _process_frame(
        self, frame: np.ndarray, frame_num: int
    ) -> Optional[List[DecisionEvent]]:
        """
        Process a single frame through the pipeline.

        Routing depends on the current state:
        - IDLE: run trigger check only (cheap).
        - ACTIVE: run detector + feed to state machine.
        """
        state = self._state_machine.state

        # Run hand tracking if enabled
        crop_roi = None
        hands = []

        if self._hand_tracker is not None:
            hands = self._hand_tracker.detect_and_track(frame)
            self._current_hands = hands

            if hands and self._config.hand_tracking.roi_crop_enabled:
                from smartbin.hand_tracker import get_hand_roi
                crop_roi = get_hand_roi(
                    frame.shape,
                    hands,
                    padding_factor=self._config.hand_tracking.roi_padding_factor,
                )

        if state == BinState.IDLE:
            # Only run the trigger — detector stays dormant
            triggered = self._trigger.check(frame)
            if triggered or (self._hand_tracker and hands):
                logger.info("Trigger fired at frame %d — activating", frame_num)
                self._state_machine.activate()
                self._detector.reset_tracker()
                self._trigger.reset()
            return None

        elif state == BinState.ACTIVE:
            # Run detection + tracking
            try:
                detections = self._detector.detect(frame, crop_roi=crop_roi)

                # Associate hands and objects if hand tracking active
                if self._hand_tracker is not None and hands:
                    from smartbin.hand_tracker import associate_hands_and_objects
                    detections = associate_hands_and_objects(
                        hands,
                        detections,
                        max_dist_px=self._config.hand_tracking.max_hand_distance_px,
                    )
                self._current_detections = detections
            except Exception:
                logger.exception(
                    "Detector error at frame %d — treating as empty", frame_num
                )
                detections = []
                self._current_detections = []

            # Feed to state machine (may trigger finalization)
            events = self._state_machine.feed(detections)
            return events

        return None

    def _fire_hooks(self, events: List[DecisionEvent]) -> None:
        """Dispatch decision events to all registered hooks."""
        for hook in self._hooks:
            try:
                hook.on_batch(events)
            except Exception:
                logger.exception("Hook %s failed", type(hook).__name__)

    def _display_frame(self, frame: np.ndarray, frame_num: int) -> None:
        """Draw state overlay, hand bboxes, and waste items on frame."""
        state = self._state_machine.state
        color = (0, 255, 0) if state == BinState.ACTIVE else (128, 128, 128)
        label = f"{state.name} | frame {frame_num}"

        if state == BinState.ACTIVE:
            label += (
                f" | buf {self._state_machine.buffer_size}"
                f"/{self._config.buffer.active_window_size}"
            )

        cv2.putText(
            frame, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )

        # Draw hands
        for hand in self._current_hands:
            hx1, hy1, hx2, hy2 = map(int, hand.bbox)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 165, 0), 2)
            cv2.putText(
                frame,
                f"Hand #{hand.hand_id}",
                (hx1, max(15, hy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 165, 0),
                2,
            )

        # Draw waste item detections
        for det in self._current_detections:
            dx1, dy1, dx2, dy2 = map(int, det.bbox)
            box_color = (0, 255, 255) if det.is_held_by_hand else (0, 255, 0)
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), box_color, 2)
            held_str = f" [Hand #{det.hand_id}]" if det.is_held_by_hand else ""
            cv2.putText(
                frame,
                f"{det.class_name} {det.confidence:.2f}{held_str}",
                (dx1, max(15, dy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2,
            )

        cv2.imshow("Smartbin", frame)

    def _shutdown(self, cap: cv2.VideoCapture) -> None:
        """Clean up resources on shutdown."""
        # Finalize any in-progress window
        events = self._state_machine.force_finalize()
        if events:
            self._fire_hooks(events)

        # Release resources
        cap.release()
        if self._show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # GUI backend unavailable — nothing to destroy

        self._detector.close()

        for hook in self._hooks:
            try:
                hook.close()
            except Exception:
                logger.exception("Hook cleanup failed for %s", type(hook).__name__)

        logger.info("Pipeline shutdown complete")
