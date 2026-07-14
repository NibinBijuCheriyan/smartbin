"""
Integration test — full pipeline cycle with a mock detector.

Verifies the end-to-end flow: trigger → state machine → detect → vote → hook,
using synthetic frames and a mock detector that returns canned detections.
No GPU, camera, or model weights required.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import List

import numpy as np
import pytest

from smartbin.config import (
    BufferConfig,
    CameraConfig,
    DisplayConfig,
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
    """Returns pre-programmed detections for testing."""

    def __init__(self, sequence: List[List[Detection]]) -> None:
        """
        Args:
            sequence: List of detection lists, one per frame.
                      Cycles if more frames are processed than sequence length.
        """
        self._sequence = sequence
        self._frame_idx = 0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self._sequence:
            return []
        detections = self._sequence[self._frame_idx % len(self._sequence)]
        self._frame_idx += 1
        return detections

    def reset_tracker(self) -> None:
        self._frame_idx = 0


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
