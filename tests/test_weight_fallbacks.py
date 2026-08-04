"""Tests for main._apply_weight_fallbacks().

Covers the edge case where neither best.pt nor yolo11n.pt exist on disk.
The function must still configure the yolo11n.pt fallback (Ultralytics
will auto-download it at model-load time), rather than calling sys.exit(1).
"""

from __future__ import annotations

from smartbin.config import ModelConfig, SmartbinConfig
from main import _apply_weight_fallbacks


class TestApplyWeightFallbacks:
    """Unit tests for _apply_weight_fallbacks()."""

    def test_fallback_when_no_weights_exist(self, tmp_path, monkeypatch):
        """Missing best.pt + no cached yolo11n.pt → sets fallback, NOT sys.exit."""
        # chdir to an empty directory so no .pt files exist on disk
        monkeypatch.chdir(tmp_path)

        config = SmartbinConfig(
            model=ModelConfig(weights="best.pt"),
        )

        result = _apply_weight_fallbacks(config)

        assert result.model.weights == "yolo11n.pt"
        assert result.allow_generic_model is True
        assert result.model.class_agnostic is True

    def test_no_fallback_when_weights_exist(self, tmp_path, monkeypatch):
        """If the configured weights file exists, no fallback occurs."""
        monkeypatch.chdir(tmp_path)
        # Create a dummy weights file
        (tmp_path / "best.pt").write_bytes(b"fake")

        config = SmartbinConfig(
            model=ModelConfig(weights="best.pt"),
        )

        result = _apply_weight_fallbacks(config)

        assert result.model.weights == "best.pt"
        assert result.allow_generic_model is False
        assert result.model.class_agnostic is False

    def test_no_fallback_for_yolo_prefixed_weights(self, tmp_path, monkeypatch):
        """Weights starting with 'yolo' skip the fallback (handled by Ultralytics)."""
        monkeypatch.chdir(tmp_path)

        config = SmartbinConfig(
            model=ModelConfig(weights="yolo11n.pt"),
        )

        result = _apply_weight_fallbacks(config)

        # Should pass through unchanged — 'yolo*' names are trusted
        assert result.model.weights == "yolo11n.pt"
        assert result.allow_generic_model is False

    def test_auto_class_agnostic_when_generic_allowed(self, tmp_path, monkeypatch):
        """allow_generic_model=True without class_agnostic → auto-enables it."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "some_model.pt").write_bytes(b"fake")

        config = SmartbinConfig(
            model=ModelConfig(weights="some_model.pt", class_agnostic=False),
            allow_generic_model=True,
        )

        result = _apply_weight_fallbacks(config)

        assert result.model.class_agnostic is True
        assert result.allow_generic_model is True
