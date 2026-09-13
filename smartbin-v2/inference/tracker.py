"""
ByteTrack spatial-temporal object tracker wrapper.
Associates detections across consecutive frames to maintain consistent track IDs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from smartbin_v2.inference.engine import DetectionResult
from smartbin_v2.utils.geometry import compute_iou
from smartbin_v2.utils.logger import get_logger

logger = get_logger("tracker")


class TrackState:
    """Represents the trajectory state of a tracked waste object."""
    def __init__(self, track_id: int, detection: DetectionResult) -> None:
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.hits = 1
        self.time_since_update = 0
        self.history: List[DetectionResult] = [detection]

    def update(self, detection: DetectionResult) -> None:
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.hits += 1
        self.time_since_update = 0
        self.history.append(detection)
        if len(self.history) > 30:
            self.history.pop(0)


class ByteTrackerWrapper:
    """
    ByteTrack implementation for multi-object tracking.
    Matches high-confidence and low-confidence detections in two association stages.
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        match_thresh: float = 0.70,
        max_time_lost: int = 25,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self._next_id = 1
        self.tracked_objects: Dict[int, TrackState] = {}

    def update(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Associate detections to existing tracks using two-stage ByteTrack matching.
        """
        # Separate high-confidence and low-confidence detections
        high_dets = [d for d in detections if d.confidence >= self.track_thresh]
        low_dets = [d for d in detections if d.confidence < self.track_thresh]

        # Age existing tracks
        for track in self.tracked_objects.values():
            track.time_since_update += 1

        # Stage 1: Associate high confidence detections with existing tracks
        unmatched_tracks: List[int] = list(self.tracked_objects.keys())
        unmatched_dets: List[DetectionResult] = []
        updated_detections: List[DetectionResult] = []

        for det in high_dets:
            best_track_id = None
            best_iou = self.match_thresh

            for tid in unmatched_tracks:
                track = self.tracked_objects[tid]
                iou = compute_iou(det.bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = tid

            if best_track_id is not None:
                self.tracked_objects[best_track_id].update(det)
                unmatched_tracks.remove(best_track_id)
                det.track_id = best_track_id
                updated_detections.append(det)
            else:
                unmatched_dets.append(det)

        # Stage 2: Associate remaining tracks with low-confidence detections
        for det in low_dets:
            best_track_id = None
            best_iou = 0.50  # Lower threshold for 2nd stage association

            for tid in unmatched_tracks:
                track = self.tracked_objects[tid]
                iou = compute_iou(det.bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = tid

            if best_track_id is not None:
                self.tracked_objects[best_track_id].update(det)
                unmatched_tracks.remove(best_track_id)
                det.track_id = best_track_id
                updated_detections.append(det)

        # Stage 3: Initiate new tracks for remaining high-confidence detections
        for det in unmatched_dets:
            new_id = self._next_id
            self._next_id += 1
            self.tracked_objects[new_id] = TrackState(new_id, det)
            det.track_id = new_id
            updated_detections.append(det)

        # Remove dead tracks that exceeded max_time_lost
        dead_tracks = [
            tid
            for tid, track in self.tracked_objects.items()
            if track.time_since_update > self.max_time_lost
        ]
        for tid in dead_tracks:
            del self.tracked_objects[tid]

        return updated_detections
