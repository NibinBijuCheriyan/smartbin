"""
Multi-Object Prioritization Engine for SmartBin AI v2.
Evaluates multiple concurrent waste items in the bin hopper and selects the single
highest priority object based on Area, Optical Center Distance, and Confidence.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from smartbin_v2.inference.engine import DetectionResult
from smartbin_v2.utils.geometry import calculate_bbox_area, calculate_optical_center_distance
from smartbin_v2.utils.logger import get_logger

logger = get_logger("prioritizer")


class MultiObjectPrioritizer:
    """
    Ranks candidate waste objects and selects the dominant object for servo segregation.
    Priority Score = w_area * Area_norm + w_dist * (1 - Dist_norm) + w_conf * Confidence
    """

    def __init__(
        self,
        weight_area: float = 0.40,
        weight_center_dist: float = 0.40,
        weight_conf: float = 0.20,
        min_area_fraction: float = 0.005,
        max_area_fraction: float = 0.85,
    ) -> None:
        self.w_area = weight_area
        self.w_dist = weight_center_dist
        self.w_conf = weight_conf
        self.min_area = min_area_fraction
        self.max_area = max_area_fraction

    def score_detection(
        self,
        detection: DetectionResult,
        frame_width: int,
        frame_height: int,
    ) -> float:
        """
        Compute weighted priority score for a single detection.
        Higher score = closer to chute center, larger physical area, and higher model confidence.
        """
        area_norm = calculate_bbox_area(detection.bbox, frame_width, frame_height, normalized=True)
        center_dist = calculate_optical_center_distance(detection.bbox, frame_width, frame_height)

        # Discard tiny noise or oversized background occlusions
        if area_norm < self.min_area or area_norm > self.max_area:
            return 0.0

        # Score formulation
        # 1 - center_dist gives highest weight to items directly in the center of the chute
        score = (
            (self.w_area * area_norm)
            + (self.w_dist * (1.0 - center_dist))
            + (self.w_conf * detection.confidence)
        )
        return float(score)

    def prioritize(
        self,
        detections: List[DetectionResult],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[Optional[DetectionResult], List[DetectionResult]]:
        """
        Score and rank all detections.
        Returns: (highest_priority_detection, ranked_all_detections)
        """
        if not detections:
            return None, []

        scored_dets: List[DetectionResult] = []
        for det in detections:
            score = self.score_detection(det, frame_width, frame_height)
            det.priority_score = score
            if score > 0.0:
                scored_dets.append(det)

        if not scored_dets:
            return None, []

        # Sort descending by priority score
        scored_dets.sort(key=lambda d: d.priority_score, reverse=True)
        primary_selection = scored_dets[0]

        logger.debug(
            f"Prioritized: {primary_selection.class_name} (Score: {primary_selection.priority_score:.3f})"
        )
        return primary_selection, scored_dets
