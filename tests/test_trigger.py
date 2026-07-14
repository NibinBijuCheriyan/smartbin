"""
Tests for the trigger gate module.

Tests use synthetic numpy frames (no camera required) to verify that
the FrameDiffTrigger fires correctly on motion and stays silent on
static scenes.
"""

from __future__ import annotations

import numpy as np
import pytest

from smartbin.config import TriggerConfig
from smartbin.trigger import FrameDiffTrigger, create_trigger


def _make_frame(h: int = 480, w: int = 640, value: int = 128) -> np.ndarray:
    """Create a uniform BGR frame."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _make_frame_with_block(
    h: int = 480,
    w: int = 640,
    bg_value: int = 128,
    block_value: int = 255,
    block_frac: float = 0.1,
) -> np.ndarray:
    """Create a frame with a bright block in the center (simulates motion)."""
    frame = np.full((h, w, 3), bg_value, dtype=np.uint8)
    bh = int(h * block_frac**0.5)
    bw = int(w * block_frac**0.5)
    cy, cx = h // 2, w // 2
    y1, y2 = cy - bh // 2, cy + bh // 2
    x1, x2 = cx - bw // 2, cx + bw // 2
    frame[y1:y2, x1:x2] = block_value
    return frame


class TestFrameDiffTrigger:
    """Tests for FrameDiffTrigger."""

    def _default_config(self, **overrides) -> TriggerConfig:
        defaults = dict(
            method="frame_diff",
            motion_threshold=25.0,
            area_fraction=0.005,
            roi=None,
            background_alpha=0.05,
        )
        defaults.update(overrides)
        return TriggerConfig(**defaults)

    def test_first_frame_never_triggers(self):
        """The very first frame initialises the background — never triggers."""
        trigger = FrameDiffTrigger(self._default_config())
        frame = _make_frame()
        assert trigger.check(frame) is False

    def test_static_scene_no_trigger(self):
        """Identical frames should not trigger."""
        trigger = FrameDiffTrigger(self._default_config())
        frame = _make_frame(value=128)

        trigger.check(frame)  # Init background
        assert trigger.check(frame) is False
        assert trigger.check(frame) is False

    def test_motion_triggers(self):
        """A large brightness change should trigger."""
        trigger = FrameDiffTrigger(self._default_config())

        bg = _make_frame(value=50)
        trigger.check(bg)  # Init background

        # Introduce a large bright block (simulates an item/hand)
        motion = _make_frame_with_block(bg_value=50, block_value=220, block_frac=0.05)
        assert trigger.check(motion) is True

    def test_small_motion_no_trigger(self):
        """Motion below area_fraction threshold should not trigger."""
        # Set a high area fraction requirement
        trigger = FrameDiffTrigger(self._default_config(area_fraction=0.5))

        bg = _make_frame(value=50)
        trigger.check(bg)

        # Small bright block (< 50% of frame)
        motion = _make_frame_with_block(bg_value=50, block_value=220, block_frac=0.01)
        assert trigger.check(motion) is False

    def test_reset_clears_background(self):
        """After reset, the next frame should re-initialise the background."""
        trigger = FrameDiffTrigger(self._default_config())

        trigger.check(_make_frame(value=100))  # Init
        trigger.reset()

        # After reset, next call should init again (never triggers)
        assert trigger.check(_make_frame(value=200)) is False

    def test_roi_crop(self):
        """Trigger should only analyse the ROI region."""
        # ROI covers a small area; motion outside ROI should be ignored
        roi = [200, 200, 400, 400]
        trigger = FrameDiffTrigger(self._default_config(roi=roi))

        bg = _make_frame(value=50)
        trigger.check(bg)

        # Put motion OUTSIDE the ROI (top-left corner)
        motion = bg.copy()
        motion[0:50, 0:50] = 255  # Outside [200:400, 200:400]
        assert trigger.check(motion) is False

        # Put motion INSIDE the ROI
        motion2 = bg.copy()
        motion2[250:350, 250:350] = 255  # Inside [200:400, 200:400]
        assert trigger.check(motion2) is True


class TestCreateTrigger:
    """Tests for the trigger factory function."""

    def test_frame_diff_creates_correctly(self):
        config = TriggerConfig(method="frame_diff")
        trigger = create_trigger(config)
        assert isinstance(trigger, FrameDiffTrigger)

    def test_unknown_method_raises(self):
        config = TriggerConfig(method="ultrasonic")
        with pytest.raises(ValueError, match="Unknown trigger method"):
            create_trigger(config)
