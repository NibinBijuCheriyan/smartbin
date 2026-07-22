"""
Hand tracking and hand-held object association module for Smartbin.

Provides:
- HandDetection: Dataclass representing a tracked hand in a video frame.
- HandTracker: Real-time hand detector & multi-object hand tracker. Uses skin color & contour analysis
  supplemented by centroid tracking to track hands across frames efficiently without requiring heavy neural networks.
- Spatial association: Links waste object detections with nearby hand track IDs.
- ROI calculation: Computes bounding boxes around hands with padding for focused, efficient YOLO inference.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HandDetection:
    """A tracked hand in a frame."""

    hand_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]  # (cx, cy)


class HandTracker:
    """
    Lightweight, fast hand detector and centroid tracker.

    Tracks hands across frames to assign stable hand_ids and provides ROI boundaries.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        max_disappeared: int = 15,
        max_distance_px: float = 120.0,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_disappeared = max_disappeared
        self.max_distance_px = max_distance_px

        self._next_hand_id = 1
        self._tracked_hands: dict[int, Tuple[Tuple[float, float, float, float], Tuple[float, float]]] = {}  # hand_id -> (bbox, center)
        self._disappeared: dict[int, int] = {}

    def detect_and_track(self, frame: np.ndarray) -> List[HandDetection]:
        """
        Detect hands in the frame and update tracking IDs.
        """
        raw_bboxes = self._detect_hand_contours(frame)

        if not raw_bboxes:
            # Mark all existing as disappeared
            for hand_id in list(self._disappeared.keys()):
                self._disappeared[hand_id] += 1
                if self._disappeared[hand_id] > self.max_disappeared:
                    self._deregister(hand_id)
            return self._to_hand_detections()

        # Input centers
        input_centers = []
        for bbox in raw_bboxes:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            input_centers.append((cx, cy))

        if len(self._tracked_hands) == 0:
            for i, bbox in enumerate(raw_bboxes):
                self._register(bbox, input_centers[i])
        else:
            # Match existing tracked hands with new detections by centroid distance
            tracked_ids = list(self._tracked_hands.keys())
            tracked_centers = [self._tracked_hands[hid][1] for hid in tracked_ids]

            # Compute distance matrix
            D = np.zeros((len(tracked_centers), len(input_centers)), dtype=np.float32)
            for i, tc in enumerate(tracked_centers):
                for j, ic in enumerate(input_centers):
                    D[i, j] = math.hypot(tc[0] - ic[0], tc[1] - ic[1])

            # Greedy assignment
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue

                if D[r, c] > self.max_distance_px:
                    continue

                hand_id = tracked_ids[r]
                self._tracked_hands[hand_id] = (raw_bboxes[c], input_centers[c])
                self._disappeared[hand_id] = 0

                used_rows.add(r)
                used_cols.add(c)

            # Unused tracked hands
            unused_rows = set(range(0, D.shape[0])) - used_rows
            for r in unused_rows:
                hand_id = tracked_ids[r]
                self._disappeared[hand_id] += 1
                if self._disappeared[hand_id] > self.max_disappeared:
                    self._deregister(hand_id)

            # Unused new detections -> register
            unused_cols = set(range(0, D.shape[1])) - used_cols
            for c in unused_cols:
                self._register(raw_bboxes[c], input_centers[c])

        return self._to_hand_detections()

    def _register(self, bbox: Tuple[float, float, float, float], center: Tuple[float, float]) -> None:
        self._tracked_hands[self._next_hand_id] = (bbox, center)
        self._disappeared[self._next_hand_id] = 0
        self._next_hand_id += 1

    def _deregister(self, hand_id: int) -> None:
        del self._tracked_hands[hand_id]
        del self._disappeared[hand_id]

    def _to_hand_detections(self) -> List[HandDetection]:
        results = []
        for hand_id, (bbox, center) in self._tracked_hands.items():
            results.append(
                HandDetection(
                    hand_id=hand_id,
                    confidence=0.85,
                    bbox=bbox,
                    center=center,
                )
            )
        return results

    def _detect_hand_contours(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """
        Fast skin-tone and region proposal contour detection for hand regions.
        """
        h, w = frame.shape[:2]
        # Convert to HSV and YCrCb for robust skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # HSV skin range
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # YCrCb skin range
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # Combine skin masks
        combined = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        # Morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.erode(combined, kernel, iterations=1)
        combined = cv2.dilate(combined, kernel, iterations=2)
        combined = cv2.GaussianBlur(combined, (5, 5), 0)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        min_area = (h * w) * 0.005  # At least 0.5% of frame area
        max_area = (h * w) * 0.35   # At most 35% of frame area

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Filter out extreme aspect ratios
                aspect_ratio = float(bw) / bh if bh > 0 else 0
                if 0.25 <= aspect_ratio <= 3.5:
                    bboxes.append((float(x), float(y), float(x + bw), float(y + bh)))

        return bboxes

    def reset(self) -> None:
        """Reset hand tracking state."""
        self._tracked_hands.clear()
        self._disappeared.clear()
        self._next_hand_id = 1


def associate_hands_and_objects(
    hands: List[HandDetection],
    detections: list,  # List of Detection objects
    max_dist_px: float = 150.0,
) -> list:
    """
    Associate waste object detections with nearby hand detections.

    Updates detection objects with hand_id and is_held_by_hand flags.
    """
    if not hands or not detections:
        return detections

    updated_detections = []
    for det in detections:
        # Calculate object center
        ox1, oy1, ox2, oy2 = det.bbox
        ocx = (ox1 + ox2) / 2.0
        ocy = (oy1 + oy2) / 2.0

        best_hand_id = None
        min_dist = float("inf")

        for hand in hands:
            hx1, hy1, hx2, hy2 = hand.bbox
            hcx, hcy = hand.center

            # Check bounding box overlap/intersection
            overlap_x = max(0.0, min(ox2, hx2) - max(ox1, hx1))
            overlap_y = max(0.0, min(oy2, hy2) - max(oy1, hy1))
            intersection = overlap_x * overlap_y

            # Center-to-center distance
            dist = math.hypot(ocx - hcx, ocy - hcy)

            # If there's direct bbox overlap or within distance threshold
            if intersection > 0 or dist <= max_dist_px:
                if dist < min_dist:
                    min_dist = dist
                    best_hand_id = hand.hand_id

        if best_hand_id is not None:
            # Import Detection type dynamically or copy fields
            from smartbin.detector import Detection
            det_updated = Detection(
                track_id=det.track_id,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                hand_id=best_hand_id,
                is_held_by_hand=True,
            )
            updated_detections.append(det_updated)
        else:
            updated_detections.append(det)

    return updated_detections


def get_hand_roi(
    frame_shape: Tuple[int, int],
    hands: List[HandDetection],
    padding_factor: float = 1.4,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute a cropped bounding box (x1, y1, x2, y2) enclosing all detected hands with padding.

    Returns None if no hands are present.
    """
    if not hands:
        return None

    h, w = frame_shape[:2]

    min_x = min(hand.bbox[0] for hand in hands)
    min_y = min(hand.bbox[1] for hand in hands)
    max_x = max(hand.bbox[2] for hand in hands)
    max_y = max(hand.bbox[3] for hand in hands)

    bw = max_x - min_x
    bh = max_y - min_y

    pad_w = (bw * (padding_factor - 1.0)) / 2.0
    pad_h = (bh * (padding_factor - 1.0)) / 2.0

    x1 = max(0, int(min_x - pad_w))
    y1 = max(0, int(min_y - pad_h))
    x2 = min(w, int(max_x + pad_w))
    y2 = min(h, int(max_y + pad_h))

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None

    return (x1, y1, x2, y2)
