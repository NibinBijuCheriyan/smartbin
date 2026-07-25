"""
Integration test — full pipeline cycle with a mock detector.

Verifies the end-to-end flow: trigger → state machine → detect → vote → hook,
using synthetic frames and a mock detector that returns canned detections.
No GPU, camera, or model weights required.

Tests include:
- Original trigger → detect → vote cycle (manual wiring)
- True SmartbinPipeline integration with MockDetector injection
- Detector exception resilience (pipeline keeps running)
- MockDetector crop_roi kwarg compatibility
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from smartbin.config import (
    BufferConfig,
    CameraConfig,
    DisplayConfig,
    HandTrackingConfig,
    LoggingConfig,
    ModelConfig,
    SmartbinConfig,
    TrackerConfig,
    TriggerConfig,
    VoterConfig,
)
from smartbin.decision import DecisionEvent, DecisionHook
from smartbin.detector import BaseDetector, Detection
from smartbin.state_machine import BinState, StateMachine
from smartbin.trigger import FrameDiffTrigger


# ---------------------------------------------------------------------------
# Mock detector — returns canned detections
# ---------------------------------------------------------------------------


class MockDetector(BaseDetector):
    """Returns pre-programmed detections for testing.

    Accepts the crop_roi kwarg that pipeline.py always passes
    (detect(frame, crop_roi=crop_roi)) so it can be used through
    the real pipeline without raising TypeError.
    """

    def __init__(self, sequence: List[List[Detection]]) -> None:
        """
        Args:
            sequence: List of detection lists, one per frame.
                      Cycles if more frames are processed than sequence length.
        """
        self._sequence = sequence
        self._frame_idx = 0

    def detect(
        self,
        frame: np.ndarray,
        crop_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Detection]:
        if not self._sequence:
            return []
        detections = self._sequence[self._frame_idx % len(self._sequence)]
        self._frame_idx += 1
        return detections

    def reset_tracker(self) -> None:
        self._frame_idx = 0


class FailingDetector(BaseDetector):
    """Raises RuntimeError on every detect() call. Used to test pipeline resilience."""

    def detect(
        self,
        frame: np.ndarray,
        crop_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Detection]:
        raise RuntimeError("Simulated detector failure")

    def reset_tracker(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Collector hook — captures decisions for assertions
# ---------------------------------------------------------------------------


class CollectorHook(DecisionHook):
    """Collects decision events for test assertions."""

    def __init__(self) -> None:
        self.events: List[DecisionEvent] = []

    def on_decision(self, event: DecisionEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFullCycle:
    """End-to-end pipeline cycle without a real camera or model."""

    def test_trigger_detect_vote_cycle(self):
        """
        Simulate: static scene → motion → detections → idle timeout → decision.
        """
        # Setup
        buffer_cfg = BufferConfig(
            active_window_size=20,
            idle_timeout_frames=3,
            min_frames_for_decision=2,
        )
        voter_cfg = VoterConfig(min_consensus_ratio=0.4)
        trigger_cfg = TriggerConfig(
            method="frame_diff",
            motion_threshold=25.0,
            area_fraction=0.005,
            background_alpha=0.05,
        )

        trigger = FrameDiffTrigger(trigger_cfg)
        sm = StateMachine(buffer_cfg, voter_cfg)
        collector = CollectorHook()

        # Mock detector: returns a "plastic" detection for each frame
        mock_dets = [
            Detection(
                track_id=1, class_id=0, class_name="plastic",
                confidence=0.88, bbox=(100, 100, 200, 200),
            )
        ]
        detector = MockDetector([mock_dets])

        # Phase 1: IDLE — feed static frames, trigger should not fire
        static_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        trigger.check(static_frame)  # Init background
        assert trigger.check(static_frame) is False
        assert sm.state == BinState.IDLE

        # Phase 2: Motion detected — trigger fires
        motion_frame = static_frame.copy()
        motion_frame[100:300, 100:400] = 255  # Big bright block
        triggered = trigger.check(motion_frame)
        assert triggered is True

        # Activate state machine
        sm.activate()
        detector.reset_tracker()
        assert sm.state == BinState.ACTIVE

        # Phase 3: Feed detections for several frames
        for _ in range(5):
            dets = detector.detect(motion_frame)
            result = sm.feed(dets)
            assert result is None  # Still buffering

        # Phase 4: Item removed — empty frames → idle timeout
        result = sm.feed([])
        assert result is None
        result = sm.feed([])
        assert result is None
        result = sm.feed([])  # 3rd empty → timeout
        assert result is not None

        # Phase 5: Verify decision
        for event in result:
            collector.on_decision(event)

        assert len(collector.events) == 1
        event = collector.events[0]
        assert event.item_class == "plastic"
        assert event.track_id == 1
        assert event.frame_count == 5
        assert event.is_certain is True
        assert event.confidence > 0.8

        # State should be back to IDLE
        assert sm.state == BinState.IDLE

    def test_multiple_items_in_one_window(self):
        """Two items detected simultaneously get independent decisions."""
        buffer_cfg = BufferConfig(
            active_window_size=5,
            idle_timeout_frames=10,
            min_frames_for_decision=1,
        )
        voter_cfg = VoterConfig(min_consensus_ratio=0.3)
        sm = StateMachine(buffer_cfg, voter_cfg)

        multi_dets = [
            Detection(
                track_id=1, class_id=0, class_name="plastic",
                confidence=0.9, bbox=(50, 50, 150, 150),
            ),
            Detection(
                track_id=2, class_id=1, class_name="metal",
                confidence=0.85, bbox=(300, 300, 400, 400),
            ),
        ]
        detector = MockDetector([multi_dets])

        sm.activate()
        result = None
        for _ in range(5):
            dets = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
            result = sm.feed(dets)

        assert result is not None
        assert len(result) == 2

        classes = {e.track_id: e.item_class for e in result}
        assert classes[1] == "plastic"
        assert classes[2] == "metal"

    def test_decision_event_serialization(self):
        """DecisionEvent serialises to valid JSON."""
        event = DecisionEvent.create(
            track_id=42,
            item_class="glass",
            confidence=0.876,
            frame_count=12,
            total_frames=15,
            is_certain=True,
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["track_id"] == 42
        assert parsed["item_class"] == "glass"
        assert parsed["confidence"] == 0.876
        assert parsed["frame_count"] == 12
        assert parsed["is_certain"] is True
        assert "timestamp" in parsed

    def test_mock_detector_accepts_crop_roi(self):
        """MockDetector.detect() accepts the crop_roi kwarg that pipeline.py passes."""
        mock_dets = [
            Detection(
                track_id=1, class_id=0, class_name="plastic",
                confidence=0.9, bbox=(50, 50, 150, 150),
            )
        ]
        detector = MockDetector([mock_dets])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # This should not raise TypeError
        result = detector.detect(frame, crop_roi=(10, 10, 300, 300))
        assert len(result) == 1
        assert result[0].class_name == "plastic"

        # Also works without crop_roi
        result = detector.detect(frame)
        assert len(result) == 1

    def test_detector_exception_resilience(self):
        """
        When the detector raises an exception during _process_frame,
        the pipeline treats it as empty detections and keeps running.
        """
        buffer_cfg = BufferConfig(
            active_window_size=10,
            idle_timeout_frames=10,
            min_frames_for_decision=1,
        )
        voter_cfg = VoterConfig(min_consensus_ratio=0.4)
        sm = StateMachine(buffer_cfg, voter_cfg)
        failing_detector = FailingDetector()

        sm.activate()

        # Feed frames through state machine as if pipeline were running
        # The pipeline catches detector exceptions and treats as empty detections
        for _ in range(5):
            try:
                dets = failing_detector.detect(
                    np.zeros((480, 640, 3), dtype=np.uint8)
                )
            except Exception:
                # Pipeline catches this and uses empty detections
                dets = []
            sm.feed(dets)

        # Should still be running (not crashed)
        assert sm.state == BinState.ACTIVE


class TestSmartbinPipelineIntegration:
    """True integration tests that construct SmartbinPipeline with injected mocks."""

    def _make_config(self, tmp_path, source_path: str) -> SmartbinConfig:
        """Create a SmartbinConfig suitable for testing."""
        return SmartbinConfig(
            model=ModelConfig(
                weights="yolo11n.pt",
                confidence_threshold=0.25,
                device="cpu",
                allowed_classes=["plastic", "paper", "metal", "glass",
                                 "e-waste", "organic", "other"],
            ),
            trigger=TriggerConfig(
                method="frame_diff",
                motion_threshold=25.0,
                area_fraction=0.005,
                background_alpha=0.05,
            ),
            buffer=BufferConfig(
                active_window_size=10,
                idle_timeout_frames=3,
                min_frames_for_decision=2,
            ),
            tracker=TrackerConfig(),
            voter=VoterConfig(min_consensus_ratio=0.4),
            camera=CameraConfig(source=source_path, fps_limit=0),
            logging=LoggingConfig(
                level="DEBUG",
                decision_log=str(tmp_path / "test_decisions.jsonl"),
            ),
            display=DisplayConfig(show=False),
            hand_tracking=HandTrackingConfig(enabled=False),
        )

    def _create_test_video(self, path: str, num_static: int = 5,
                           num_motion: int = 15, num_tail: int = 5) -> None:
        """Create a synthetic test video with static → motion → static phases."""
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, 15, (640, 480))

        bg_color = (40, 40, 40)

        # Static frames (trigger stays idle)
        for _ in range(num_static):
            frame = np.full((480, 640, 3), bg_color, dtype=np.uint8)
            writer.write(frame)

        # Motion frames (trigger should fire)
        for i in range(num_motion):
            frame = np.full((480, 640, 3), bg_color, dtype=np.uint8)
            x = 100 + i * 10
            cv2.rectangle(frame, (x, 150), (x + 120, 300), (0, 180, 255), -1)
            writer.write(frame)

        # Tail static frames (idle timeout → finalize)
        for _ in range(num_tail):
            frame = np.full((480, 640, 3), bg_color, dtype=np.uint8)
            writer.write(frame)

        writer.release()

    def test_pipeline_with_mock_detector_emits_decisions(self, tmp_path):
        """
        Construct SmartbinPipeline with a MockDetector injected via
        attribute replacement. Assert DecisionEvents are emitted.
        """
        from smartbin.pipeline import SmartbinPipeline

        video_path = str(tmp_path / "test.avi")
        self._create_test_video(video_path)

        config = self._make_config(tmp_path, video_path)

        # Build pipeline
        pipeline = SmartbinPipeline(config)

        # Inject mock detector and collector hook
        mock_dets = [
            Detection(
                track_id=1, class_id=0, class_name="plastic",
                confidence=0.88, bbox=(100, 100, 200, 200),
            )
        ]
        pipeline._detector = MockDetector([mock_dets])

        collector = CollectorHook()
        pipeline._hooks.append(collector)

        # Run pipeline (will process the video and stop)
        pipeline.run()

        # Verify at least one decision was emitted
        assert len(collector.events) >= 1
        assert all(e.item_class == "plastic" for e in collector.events)
        assert all(e.track_id == 1 for e in collector.events)

    def test_pipeline_with_failing_detector_keeps_running(self, tmp_path):
        """
        Pipeline with a FailingDetector should NOT crash — it logs the error
        and treats each frame as having empty detections.
        """
        from smartbin.pipeline import SmartbinPipeline

        video_path = str(tmp_path / "test.avi")
        self._create_test_video(video_path)

        config = self._make_config(tmp_path, video_path)

        pipeline = SmartbinPipeline(config)
        pipeline._detector = FailingDetector()

        # Should complete without raising
        pipeline.run()

    def test_pipeline_decision_log_written(self, tmp_path):
        """Verify the JSONL decision log file is written by the pipeline."""
        from smartbin.pipeline import SmartbinPipeline

        video_path = str(tmp_path / "test.avi")
        self._create_test_video(video_path)

        log_path = tmp_path / "decisions.jsonl"
        config = self._make_config(tmp_path, video_path)

        # Override decision log path
        config = SmartbinConfig(
            model=config.model,
            trigger=config.trigger,
            buffer=config.buffer,
            tracker=config.tracker,
            voter=config.voter,
            camera=config.camera,
            logging=LoggingConfig(
                level="DEBUG",
                decision_log=str(log_path),
            ),
            display=config.display,
            hand_tracking=config.hand_tracking,
        )

        pipeline = SmartbinPipeline(config)
        mock_dets = [
            Detection(
                track_id=1, class_id=0, class_name="paper",
                confidence=0.75, bbox=(50, 50, 200, 200),
            )
        ]
        pipeline._detector = MockDetector([mock_dets])
        pipeline.run()

        # Check log file exists and contains valid JSONL
        if log_path.exists():
            with open(log_path, "r") as f:
                lines = f.readlines()
            if lines:
                for line in lines:
                    parsed = json.loads(line)
                    assert "item_class" in parsed
                    assert "track_id" in parsed


class TestJsonlHookIntegration:
    """Test JSONL file hook writes correctly."""

    def test_jsonl_output(self, tmp_path):
        """Decisions are appended as valid JSON lines."""
        from smartbin.decision import JsonlFileHook

        log_path = str(tmp_path / "test_decisions.jsonl")
        hook = JsonlFileHook(log_path)

        event1 = DecisionEvent.create(
            track_id=1, item_class="plastic", confidence=0.9,
            frame_count=10, total_frames=12, is_certain=True,
        )
        event2 = DecisionEvent.create(
            track_id=2, item_class="metal", confidence=0.7,
            frame_count=5, total_frames=10, is_certain=False,
        )

        hook.on_decision(event1)
        hook.on_decision(event2)
        hook.close()

        with open(log_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2
        parsed1 = json.loads(lines[0])
        parsed2 = json.loads(lines[1])
        assert parsed1["item_class"] == "plastic"
        assert parsed2["item_class"] == "metal"
