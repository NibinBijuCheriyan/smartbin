"""
Second-stage waste classifier using EfficientNet-B0 TFLite model.

Takes bounding-box crops from the YOLO detector and runs full-frame
classification to refine/override YOLO's predicted label with the higher-accuracy
EfficientNet-B0 classifier (96.59% accuracy).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from smartbin.config import RefinerConfig

logger = logging.getLogger(__name__)


# List of classes that the EfficientNet classifier does NOT support.
# If YOLO predicts one of these, the refiner should be skipped.
UNSUPPORTED_REFINER_CLASSES = {"glass", "e-waste"}


class WasteClassifier:
    """
    EfficientNet-B0 TFLite second-stage classifier.

    Preprocesses image crops to 224x224 RGB float32 arrays and performs
    inference using TFLite runtime.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        classes_path: Union[str, Path],
        confidence_threshold: float = 0.25,
        interpreter=None,
    ) -> None:
        self.model_path = Path(model_path)
        self.classes_path = Path(classes_path)
        self.confidence_threshold = confidence_threshold
        self._interpreter = interpreter
        self._classes: List[str] = []
        self._idx_to_class: dict[str, str] = {}

        self._load_classes()

    def _load_classes(self) -> None:
        """Load class index mapping from classes.json."""
        if not self.classes_path.exists():
            logger.warning("Classes file not found at %s", self.classes_path)
            # Default 5 classes if file missing
            self._classes = ["plastic", "paper", "metal", "organic_waste", "none"]
            self._idx_to_class = {str(i): cls for i, cls in enumerate(self._classes)}
            return

        try:
            with open(self.classes_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "idx_to_class" in data:
                self._idx_to_class = {str(k): str(v) for k, v in data["idx_to_class"].items()}
                self._classes = [self._idx_to_class.get(str(i), "") for i in range(len(self._idx_to_class))]
            elif "class_names" in data:
                self._classes = data["class_names"]
                self._idx_to_class = {str(i): name for i, name in enumerate(self._classes)}
        except Exception as e:
            logger.error("Failed to load classes from %s: %s", self.classes_path, e)
            self._classes = ["plastic", "paper", "metal", "organic_waste", "none"]
            self._idx_to_class = {str(i): cls for i, cls in enumerate(self._classes)}

    def _ensure_loaded(self) -> None:
        """Lazy-load the TFLite interpreter."""
        if self._interpreter is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model weights not found at {self.model_path}")

        try:
            import ai_edge_litert.interpreter as tflite
            self._interpreter = tflite.Interpreter(model_path=str(self.model_path))
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                self._interpreter = tflite.Interpreter(model_path=str(self.model_path))
            except ImportError:
                try:
                    import tensorflow.lite as tflite
                    self._interpreter = tflite.Interpreter(model_path=str(self.model_path))
                except ImportError as e:
                    raise ImportError(
                        "Could not import TFLite runtime ('ai-edge-litert', 'tflite-runtime', or 'tensorflow.lite'). "
                        "Please install ai-edge-litert or tflite-runtime or tensorflow."
                    ) from e

        self._interpreter.allocate_tensors()
        logger.info("Loaded TFLite WasteClassifier model from %s", self.model_path)

    def preprocess(self, crop: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess BGR/RGB image crop for EfficientNet-B0.

        Resizes to target size (224, 224) using bilinear interpolation,
        converts to float32 numpy array, and adds batch dimension.
        """
        if crop is None or crop.size == 0:
            raise ValueError("Crop image is empty or None")

        # Convert BGR to RGB if 3-channel image
        if len(crop.shape) == 3 and crop.shape[2] == 3:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            crop_rgb = crop

        # Resize to 224x224 (bilinear)
        resized = cv2.resize(crop_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        img_array = np.array(resized, dtype=np.float32)
        # Expand dimensions to (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single image crop.

        Args:
            crop: BGR image crop (np.ndarray).

        Returns:
            Tuple of (predicted_class_name, confidence_float).
        """
        if crop is None or crop.size == 0:
            return ("none", 0.0)

        self._ensure_loaded()

        input_data = self.preprocess(crop)

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        expected_dtype = input_details[0]["dtype"]
        self._interpreter.set_tensor(input_details[0]["index"], input_data.astype(expected_dtype))
        self._interpreter.invoke()

        output_data = self._interpreter.get_tensor(output_details[0]["index"])[0]

        # Softmax if output raw logits, or read probabilities
        if np.max(output_data) > 1.0 or np.min(output_data) < 0.0:
            # Apply softmax for raw logits
            exp_data = np.exp(output_data - np.max(output_data))
            probs = exp_data / np.sum(exp_data)
        else:
            probs = output_data

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        predicted_class = self._idx_to_class.get(str(predicted_idx), f"class_{predicted_idx}")

        return (predicted_class, confidence)


def create_refiner(config: RefinerConfig) -> Optional[WasteClassifier]:
    """Factory function to instantiate WasteClassifier if enabled in config."""
    if not config.enabled:
        logger.info("Second-stage waste classifier disabled in configuration")
        return None

    try:
        classifier = WasteClassifier(
            model_path=config.model_path,
            classes_path=config.classes_path,
            confidence_threshold=config.confidence_threshold,
        )
        return classifier
    except Exception as e:
        logger.warning("Failed to initialize WasteClassifier: %s", e)
        return None
