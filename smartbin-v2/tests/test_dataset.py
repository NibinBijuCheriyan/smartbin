"""
Unit tests for dataset tools, geometry transformations, and taxonomy validation.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from smartbin_v2.dataset_builder.deduplicator import compute_dhash, hamming_distance
from smartbin_v2.utils.geometry import (
    calculate_bbox_area,
    calculate_optical_center_distance,
    compute_iou,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_taxonomy_schema():
    """Verify Indian waste taxonomy contains required structure and valid categories."""
    tax_path = Path("smartbin-v2/datasets/indian_waste_taxonomy.json")
    assert tax_path.exists(), "Taxonomy file missing"

    with open(tax_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "classes" in data
    assert len(data["classes"]) >= 28
    for item in data["classes"]:
        assert "id" in item
        assert "name" in item
        assert "target_bin" in item
        assert item["target_bin"] in ["recyclable", "compost", "landfill", "reject"]


def test_geometry_conversions():
    """Verify coordinate transformations between xyxy and xywh."""
    bbox_xyxy = (100.0, 150.0, 300.0, 450.0)
    cx, cy, w, h = xyxy_to_xywh(bbox_xyxy)
    assert cx == 200.0
    assert cy == 300.0
    assert w == 200.0
    assert h == 300.0

    recovered = xywh_to_xyxy((cx, cy, w, h))
    assert recovered == bbox_xyxy


def test_iou_calculation():
    """Verify IoU between overlapping and non-overlapping boxes."""
    b1 = (0.0, 0.0, 10.0, 10.0)
    b2 = (5.0, 0.0, 15.0, 10.0)
    iou = compute_iou(b1, b2)
    # Intersection = 5*10 = 50, Union = 100 + 100 - 50 = 150 -> IoU = 50/150 = 1/3
    assert abs(iou - (1.0 / 3.0)) < 1e-4

    b3 = (20.0, 20.0, 30.0, 30.0)
    assert compute_iou(b1, b3) == 0.0


def test_optical_center_distance():
    """Verify normalized center distance logic."""
    frame_w, frame_h = 640, 640
    # Perfect center box
    center_box = (310.0, 310.0, 330.0, 330.0)
    dist = calculate_optical_center_distance(center_box, frame_w, frame_h)
    assert dist < 0.01

    # Corner box
    corner_box = (0.0, 0.0, 20.0, 20.0)
    dist_corner = calculate_optical_center_distance(corner_box, frame_w, frame_h)
    assert dist_corner > 0.85


def test_deduplicator_dhash():
    """Verify difference hashing yields 0 distance for identical images."""
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
    img2 = img1.copy()
    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    assert hamming_distance(h1, h2) == 0
