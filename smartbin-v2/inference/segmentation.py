"""
YOLO11-seg Instance Segmentation Pipeline for SmartBin AI v2.
Extracts instance masks, computes exact object pixel area, and performs background removal.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np

from smartbin_v2.inference.engine import DetectionResult
from smartbin_v2.utils.logger import get_logger

logger = get_logger("segmentation")


class WasteSegmentationProcessor:
    """
    Processes polygon instance masks for precise waste volume estimation and background removal.
    """

    def __init__(self, mask_threshold: float = 0.50) -> None:
        self.mask_threshold = mask_threshold

    def compute_exact_pixel_area(self, mask: np.ndarray) -> int:
        """Calculate total number of positive foreground object pixels."""
        if mask is None:
            return 0
        return int(np.sum(mask > self.mask_threshold))

    def extract_contour_polygon(self, mask: np.ndarray) -> List[np.ndarray]:
        """Extract polygon contours from binary mask for edge visualization."""
        if mask is None:
            return []
        binary = (mask > self.mask_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def remove_background(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        return_transparent_png: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Segment foreground waste item from the bin hopper, masking out the background.
        If return_transparent_png is True, returns 4-channel BGRA image.
        """
        if detection.mask is None:
            return None

        h, w = frame.shape[:2]
        # Resize mask to frame dimensions if necessary
        mask = detection.mask
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        binary_mask = (mask > self.mask_threshold).astype(np.uint8)

        if return_transparent_png:
            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = binary_mask * 255
            return bgra
        else:
            # Solid black background
            masked = frame.copy()
            masked[binary_mask == 0] = 0
            return masked

    def overlay_masks(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        alpha: float = 0.40,
    ) -> np.ndarray:
        """
        Render semi-transparent colored masks and bounding contours over the image.
        """
        out = frame.copy()
        h, w = frame.shape[:2]

        color_palette = [
            (46, 204, 113),  # Green (Compost)
            (52, 152, 219),  # Blue (Recyclable)
            (231, 76, 60),   # Red (Landfill)
            (241, 196, 15),  # Yellow
            (155, 89, 182),  # Purple
        ]

        for det in detections:
            if det.mask is None:
                continue

            mask = det.mask
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

            binary = (mask > self.mask_threshold).astype(bool)
            color = color_palette[det.class_id % len(color_palette)]

            # Blend color
            overlay = out.copy()
            overlay[binary] = color
            out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)

            # Draw boundary contour
            contours = self.extract_contour_polygon(det.mask)
            cv2.drawContours(out, contours, -1, color, 2)

        return out
