"""
Hand tracking and hand-held object association module for Smartbin.

Provides:
- HandDetection: Dataclass representing a tracked hand in a video frame.
- MediaPipeHandTracker: Production hand detector using MediaPipe Hands (supports both
  legacy solutions API and modern Tasks API with automatic fallback).
  Reliable across skin tones, lighting conditions, and gloved hands.
- SkinColorHandTracker: Legacy fallback using HSV/YCrCb skin-tone thresholding.
  KNOWN LIMITATIONS (see class docstring). Only for constrained hardware
  where MediaPipe is too heavy.
- Spatial association: Links waste object detections with nearby hand track IDs.
- ROI calculation: Computes bounding boxes around hands with padding for
  focused, efficient YOLO inference.
- Factory function: create_hand_tracker() selects backend based on config.
"""

from __future__ import annotations

import math
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Base centroid tracker mixin (shared by both backends)
# ---------------------------------------------------------------------------


class _CentroidTrackerMixin:
    """
    Centroid-based multi-object tracker for assigning stable hand IDs.

    Both MediaPipe and skin-color backends produce per-frame bounding boxes.
    This mixin associates them across frames using centroid distance matching
    to provide stable hand_id values.
    """

    def _init_tracker(
        self,
        max_disappeared: int = 15,
        max_distance_px: float = 120.0,
    ) -> None:
        self._max_disappeared = max_disappeared
        self._max_distance_px = max_distance_px
        self._next_hand_id = 1
        self._tracked_hands: dict[int, Tuple[Tuple[float, float, float, float], Tuple[float, float]]] = {}
        self._disappeared: dict[int, int] = {}

    def _update_tracks(
        self, raw_bboxes: List[Tuple[float, float, float, float]]
    ) -> List[HandDetection]:
        """Update tracking state with new detections and return tracked hands."""
        if not raw_bboxes:
            for hand_id in list(self._disappeared.keys()):
                self._disappeared[hand_id] += 1
                if self._disappeared[hand_id] > self._max_disappeared:
                    self._deregister(hand_id)
            return self._to_hand_detections()

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
            tracked_ids = list(self._tracked_hands.keys())
            tracked_centers = [self._tracked_hands[hid][1] for hid in tracked_ids]

            D = np.zeros((len(tracked_centers), len(input_centers)), dtype=np.float32)
            for i, tc in enumerate(tracked_centers):
                for j, ic in enumerate(input_centers):
                    D[i, j] = math.hypot(tc[0] - ic[0], tc[1] - ic[1])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows: set = set()
            used_cols: set = set()

            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                if D[r, c] > self._max_distance_px:
                    continue

                hand_id = tracked_ids[r]
                self._tracked_hands[hand_id] = (raw_bboxes[c], input_centers[c])
                self._disappeared[hand_id] = 0
                used_rows.add(r)
                used_cols.add(c)

            unused_rows = set(range(0, D.shape[0])) - used_rows
            for r in unused_rows:
                hand_id = tracked_ids[r]
                self._disappeared[hand_id] += 1
                if self._disappeared[hand_id] > self._max_disappeared:
                    self._deregister(hand_id)

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

    def reset(self) -> None:
        """Reset hand tracking state."""
        self._tracked_hands.clear()
        self._disappeared.clear()
        self._next_hand_id = 1


# ---------------------------------------------------------------------------
# MediaPipe hand tracker (production, recommended)
# ---------------------------------------------------------------------------


class MediaPipeHandTracker(_CentroidTrackerMixin):
    """
    Hand detector using Google MediaPipe.

    Supports both legacy `mediapipe.solutions.hands` (MediaPipe <0.10) and modern
    `mediapipe.tasks.python.vision.HandLandmarker` (MediaPipe >=0.10 / Python 3.13+).

    Reliable across:
    - All skin tones (uses ML-based detection, not color thresholding)
    - Varying lighting conditions (indoor, outdoor, artificial)
    - Gloved hands (detects hand shape, not skin color)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        max_disappeared: int = 15,
        max_distance_px: float = 120.0,
        max_num_hands: int = 2,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_distance_px = max_distance_px
        self._init_tracker(max_disappeared, max_distance_px)
        self._max_num_hands = max_num_hands

        # Lazy-load MediaPipe
        self._hands = None
        self._landmarker = None
        self._fallback_tracker: Optional[SkinColorHandTracker] = None

    def _ensure_loaded(self) -> None:
        """Lazy-load MediaPipe Hands on first use."""
        if self._hands is not None or self._landmarker is not None or self._fallback_tracker is not None:
            return

        try:
            import mediapipe as mp

            # 1. Try legacy solutions API (MediaPipe < 0.10)
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self._mp_hands = mp.solutions.hands
                self._hands = self._mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self._max_num_hands,
                    min_detection_confidence=self.confidence_threshold,
                    min_tracking_confidence=self.confidence_threshold,
                )
                logger.info("MediaPipe Hands (solutions API) initialized (max_hands=%d)", self._max_num_hands)
                return

            # 2. Try Tasks API (MediaPipe >= 0.10 / Python 3.13+)
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                model_path = Path("data/hand_landmarker.task")
                if not model_path.exists():
                    logger.info("Downloading MediaPipe hand_landmarker.task model...")
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                    urllib.request.urlretrieve(url, model_path)

                base_options = python.BaseOptions(model_asset_path=str(model_path))
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=self._max_num_hands,
                    min_hand_detection_confidence=self.confidence_threshold,
                    min_hand_presence_confidence=self.confidence_threshold,
                )
                self._landmarker = vision.HandLandmarker.create_from_options(options)
                logger.info("MediaPipe HandLandmarker (Tasks API) initialized from %s", model_path)
                return
            except Exception as e:
                logger.warning("Could not initialize MediaPipe Tasks API: %s", e)

        except ImportError:
            logger.warning("MediaPipe package is not installed.")

        # Fallback to SkinColorHandTracker if MediaPipe is unavailable or failed
        logger.warning("Falling back to SkinColorHandTracker due to MediaPipe load error.")
        self._fallback_tracker = SkinColorHandTracker(
            confidence_threshold=self.confidence_threshold,
            max_disappeared=self._max_disappeared,
            max_distance_px=self._max_distance_px,
        )

    def detect_and_track(self, frame: np.ndarray) -> List[HandDetection]:
        """Detect hands using MediaPipe and update tracking IDs."""
        self._ensure_loaded()

        if self._fallback_tracker is not None:
            return self._fallback_tracker.detect_and_track(frame)

        h, w = frame.shape[:2]
        raw_bboxes: List[Tuple[float, float, float, float]] = []

        if self._hands is not None:
            # Legacy solutions API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * h for lm in hand_landmarks.landmark]

                    x1 = max(0, min(x_coords))
                    y1 = max(0, min(y_coords))
                    x2 = min(w, max(x_coords))
                    y2 = min(h, max(y_coords))

                    if (x2 - x1) > 10 and (y2 - y1) > 10:
                        raw_bboxes.append((float(x1), float(y1), float(x2), float(y2)))

        elif self._landmarker is not None:
            # New Tasks API
            import mediapipe as mp
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self._landmarker.detect(mp_image)

            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    x_coords = [lm.x * w for lm in hand_landmarks]
                    y_coords = [lm.y * h for lm in hand_landmarks]

                    x1 = max(0, min(x_coords))
                    y1 = max(0, min(y_coords))
                    x2 = min(w, max(x_coords))
                    y2 = min(h, max(y_coords))

                    if (x2 - x1) > 10 and (y2 - y1) > 10:
                        raw_bboxes.append((float(x1), float(y1), float(x2), float(y2)))

        return self._update_tracks(raw_bboxes)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


# ---------------------------------------------------------------------------
# Skin-color hand tracker (legacy fallback for constrained hardware)
# ---------------------------------------------------------------------------


class SkinColorHandTracker(_CentroidTrackerMixin):
    """
    Legacy hand detector using HSV/YCrCb skin-tone thresholding.

    ⚠️  KNOWN FAILURE MODES — Use MediaPipeHandTracker instead unless
    hardware constraints make MediaPipe infeasible:

    1. SKIN TONE BIAS: HSV/YCrCb ranges are tuned for a narrow band of
       skin tones. Hands with very dark or very light skin may not be
       detected, or may require per-deployment calibration.

    2. FALSE POSITIVES: Any skin-colored object triggers detection —
       wood grain, cardboard, tan plastic, food items, and even warm-toned
       backgrounds will produce false hand detections.

    3. LIGHTING SENSITIVITY: Performance degrades significantly under
       non-standard lighting (strong blue/green tint, very warm tungsten,
       direct sunlight causing overexposure).

    4. GLOVED HANDS: Completely fails on gloved hands (surgical, work,
       winter gloves) since there is no skin color to detect.

    5. NO POSE INFORMATION: Cannot distinguish a hand from a forearm,
       face, or any other skin-colored body part.

    This tracker is preserved for deployments on extremely constrained
    hardware (e.g., bare Raspberry Pi without MediaPipe support).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        max_disappeared: int = 15,
        max_distance_px: float = 120.0,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_distance_px = max_distance_px
        self._init_tracker(max_disappeared, max_distance_px)

    def detect_and_track(self, frame: np.ndarray) -> List[HandDetection]:
        """Detect hands via skin-color thresholding and update tracking IDs."""
        raw_bboxes = self._detect_hand_contours(frame)
        return self._update_tracks(raw_bboxes)

    def _detect_hand_contours(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """
        Fast skin-tone and region proposal contour detection for hand regions.

        See class docstring for known failure modes of this approach.
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
                aspect_ratio = float(bw) / bh if bh > 0 else 0
                if 0.25 <= aspect_ratio <= 3.5:
                    bboxes.append((float(x), float(y), float(x + bw), float(y + bh)))

        return bboxes


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

HandTracker = MediaPipeHandTracker


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_hand_tracker(
    backend: str = "mediapipe",
    confidence_threshold: float = 0.3,
    max_distance_px: float = 120.0,
) -> _CentroidTrackerMixin:
    """
    Create a hand tracker with the specified backend.

    Args:
        backend: "mediapipe" (recommended) or "skin_color" (legacy fallback).
        confidence_threshold: Minimum detection confidence.
        max_distance_px: Maximum centroid distance for track association.

    Returns:
        A hand tracker instance with detect_and_track() method.
    """
    if backend == "skin_color":
        logger.warning(
            "Using SkinColorHandTracker (legacy). See class docstring for "
            "known failure modes. Consider switching to 'mediapipe' backend."
        )
        return SkinColorHandTracker(
            confidence_threshold=confidence_threshold,
            max_distance_px=max_distance_px,
        )
    elif backend == "mediapipe":
        return MediaPipeHandTracker(
            confidence_threshold=confidence_threshold,
            max_distance_px=max_distance_px,
        )
    else:
        raise ValueError(
            f"Unknown hand tracker backend: '{backend}'. "
            f"Use 'mediapipe' or 'skin_color'."
        )


# ---------------------------------------------------------------------------
# Spatial association utilities
# ---------------------------------------------------------------------------


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
