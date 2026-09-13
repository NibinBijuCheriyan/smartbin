"""
Geometry and bounding box transformation utilities.
Handles coordinate formats, normalized area computation, center distance, and IoU.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Union
import numpy as np


def xyxy_to_xywh(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return (cx, cy, w, h)


def xywh_to_xyxy(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = bbox
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return (x1, y1, x2, y2)


def clip_bbox(
    bbox: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[int, int, int, int]:
    """Clip [x1, y1, x2, y2] coordinates to image boundaries."""
    x1, y1, x2, y2 = bbox
    x1_c = int(max(0, min(x1, width - 1)))
    y1_c = int(max(0, min(y1, height - 1)))
    x2_c = int(max(x1_c + 1, min(x2, width)))
    y2_c = int(max(y1_c + 1, min(y2, height)))
    return (x1_c, y1_c, x2_c, y2_c)


def calculate_bbox_area(
    bbox: Tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    normalized: bool = True,
) -> float:
    """
    Calculate bounding box area.
    If normalized is True, returns area as a fraction of total frame area (0.0 to 1.0).
    """
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    if normalized:
        total_area = float(frame_width * frame_height)
        return min(1.0, area / total_area) if total_area > 0 else 0.0
    return area


def calculate_optical_center_distance(
    bbox: Tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> float:
    """
    Calculate normalized distance from bounding box center to optical center of frame.
    Returns value between 0.0 (exact center) and 1.0 (furthest corner).
    """
    x1, y1, x2, y2 = bbox
    box_cx = (x1 + x2) / 2.0
    box_cy = (y1 + y2) / 2.0

    frame_cx = frame_width / 2.0
    frame_cy = frame_height / 2.0

    # Euclidean distance
    dist = math.sqrt((box_cx - frame_cx) ** 2 + (box_cy - frame_cy) ** 2)

    # Maximum possible distance from center to corner
    max_dist = math.sqrt(frame_cx**2 + frame_cy**2)

    return min(1.0, dist / max_dist) if max_dist > 0 else 0.0


def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - intersection_area
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area
