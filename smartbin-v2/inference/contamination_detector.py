"""Contamination detection backed only by a trained binary classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from smartbin_v2.inference.engine import DetectionResult
from smartbin_v2.utils.geometry import clip_bbox
from smartbin_v2.utils.logger import get_logger

logger = get_logger("contamination_detector")


class ContaminationDetector:
    """Classifies clean versus contaminated items when a trained model is available."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        threshold: float = 0.65,
        reroute_to_landfill: bool = True,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.threshold = threshold
        self.reroute_to_landfill = reroute_to_landfill
        self.interpreter = None
        if self.model_path and self.model_path.exists():
            self._init_classifier()

    @property
    def is_ready(self) -> bool:
        return self.interpreter is not None

    def _init_classifier(self) -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter
            self.interpreter = Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info("Loaded contamination detector model: %s", self.model_path)
        except Exception as exc:
            logger.warning("Could not load contamination model: %s", exc)
            self.interpreter = None

    def predict(self, crop: np.ndarray) -> Tuple[bool, float]:
        if not self.is_ready or crop.size == 0:
            return False, 0.0
        resized = cv2.resize(crop, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        dirty_conf = float(output[1]) if len(output) > 1 else float(output[0])
        return dirty_conf >= self.threshold, dirty_conf

    def evaluate_detection(self, frame: np.ndarray, detection: DetectionResult) -> DetectionResult:
        if not self.is_ready:
            return detection
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = clip_bbox(detection.bbox, w, h)
        is_contaminated, confidence = self.predict(frame[y1:y2, x1:x2])
        detection.is_contaminated = is_contaminated
        if is_contaminated:
            logger.info("Contamination detected on '%s' (score: %.2f)", detection.class_name, confidence)
        return detection