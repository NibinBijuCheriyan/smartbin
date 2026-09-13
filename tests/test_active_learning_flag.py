"""Tests for the active learning flagging script.

Covers CSV schema correctness and detection filtering logic.
No model or video required — uses mocked detections.
"""

from __future__ import annotations

import csv
import os
import tempfile

import pytest

from active_learning_flag import CSV_COLUMNS, should_flag


class TestShouldFlag:
    """Tests for the should_flag filtering function."""

    def test_low_confidence_plastic_flagged(self):
        assert should_flag("plastic", 0.35) is True

    def test_high_confidence_plastic_not_flagged(self):
        assert should_flag("plastic", 0.75) is False

    def test_boundary_confidence_not_flagged(self):
        """Confidence exactly at 0.5 is NOT below threshold."""
        assert should_flag("plastic", 0.5) is False

    def test_just_below_boundary_flagged(self):
        assert should_flag("plastic", 0.499) is True

    def test_other_class_always_flagged(self):
        """'other' class is always flagged regardless of confidence."""
        assert should_flag("other", 0.99) is True
        assert should_flag("other", 0.10) is True

    def test_other_class_case_insensitive(self):
        assert should_flag("Other", 0.99) is True
        assert should_flag("OTHER", 0.99) is True

    def test_high_confidence_non_other_not_flagged(self):
        assert should_flag("paper", 0.80) is False
        assert should_flag("metal", 0.65) is False
        assert should_flag("glass", 0.55) is False


class TestCSVSchema:
    """Tests for CSV output schema."""

    def test_csv_columns_are_correct(self):
        expected = ["frame_number", "track_id", "predicted_class", "confidence", "crop_path"]
        assert CSV_COLUMNS == expected

    def test_csv_roundtrip(self):
        """Write and read back a CSV row to verify schema integrity."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerow({
                "frame_number": 42,
                "track_id": 3,
                "predicted_class": "other",
                "confidence": "0.4500",
                "crop_path": "crops/frame000042_track3_other_0.450.jpg",
            })
            tmp_path = f.name

        try:
            with open(tmp_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                assert list(reader.fieldnames) == CSV_COLUMNS
                rows = list(reader)
                assert len(rows) == 1
                row = rows[0]
                assert row["frame_number"] == "42"
                assert row["track_id"] == "3"
                assert row["predicted_class"] == "other"
                assert float(row["confidence"]) == pytest.approx(0.45)
                assert "crop" in row["crop_path"]
        finally:
            os.unlink(tmp_path)
