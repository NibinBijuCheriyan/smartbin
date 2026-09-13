"""
Second-Stage Material Classification and Multimodal Confidence Fusion.
Predicts specific material grades (PET, HDPE, LDPE, Aluminium, Steel, Glass colors, Paper grades)
and fuses material predictions with primary YOLO detection classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from smartbin_v2.inference.engine import DetectionResult
from smartbin_v2.utils.geometry import clip_bbox
from smartbin_v2.utils.logger import get_logger

logger = get_logger("material_classifier")

# 12 Detailed Material Classes
MATERIAL_CLASSES = [
    "pet_plastic",
    "hdpe_plastic",
    "ldpe_plastic",
    "aluminium_metal",
    "steel_metal",
    "clear_glass",
    "brown_glass",
    "green_glass",
    "cardboard_paper",
    "newspaper_paper",
    "carton_tetra",
    "organic_compost",
    "landfill_inert",
]

# Compatibility matrix between YOLO detected classes and expected materials
MATERIAL_COMPATIBILITY: Dict[str, List[str]] = {
    "pet_bottle": ["pet_plastic"],
    "hdpe_bottle": ["hdpe_plastic"],
    "shampoo_bottle": ["hdpe_plastic", "pet_plastic"],
    "milk_packet": ["ldpe_plastic"],
    "water_sachet": ["ldpe_plastic"],
    "plastic_carry_bag": ["ldpe_plastic", "hdpe_plastic"],
    "chips_packet": ["ldpe_plastic"],
    "kurkure_packet": ["ldpe_plastic"],
    "biscuit_wrapper": ["ldpe_plastic"],
    "aluminium_can": ["aluminium_metal"],
    "tin_can": ["steel_metal", "aluminium_metal"],
    "glass_bottle": ["clear_glass", "brown_glass", "green_glass"],
    "glass_jar": ["clear_glass", "brown_glass"],
    "cardboard": ["cardboard_paper"],
    "newspaper": ["newspaper_paper"],
    "paper_cup": ["cardboard_paper"],
    "tetra_pak": ["carton_tetra"],
    "banana_peel": ["organic_compost"],
    "coconut_shell": ["organic_compost"],
    "egg_shell": ["organic_compost"],
    "fruit_waste": ["organic_compost"],
    "vegetable_waste": ["organic_compost"],
    "rice_waste": ["organic_compost"],
    "leaves": ["organic_compost"],
    "tea_bag": ["organic_compost"],
    "styrofoam": ["landfill_inert"],
    "tissue": ["landfill_inert"],
    "mask": ["landfill_inert"],
    "cigarette_butt": ["landfill_inert"],
    "sanitary_waste": ["landfill_inert"],
    "ceramic": ["landfill_inert"],
    "broken_plastic_toys": ["landfill_inert"],
}


class MaterialClassifier:
    """
    Second-stage deep learning classifier predicting material resin / grade from bounding box crops.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        input_size: Tuple[int, int] = (224, 224),
        fusion_weight: float = 0.35,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.input_size = input_size
        self.fusion_weight = fusion_weight  # Weight for 2nd stage in fused confidence
        self.interpreter = None

        if self.model_path and self.model_path.exists():
            self._load_tflite_model()


    @property
    def is_ready(self) -> bool:
        return self.interpreter is not None
    def _load_tflite_model(self) -> None:
        """Load TFLite or LiteRT interpreter."""
        try:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter

            self.interpreter = Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(f"Loaded 2nd-stage material classifier: {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load material model ({e}); using heuristic rule-based material head.")

    def predict_material(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Infer material category and confidence from cropped waste object.
        """
        if crop.size == 0:
            return "unknown_material", 0.0

        if self.interpreter is not None:
            # Resize and normalize
            resized = cv2.resize(crop, self.input_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)

            self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

            cls_idx = int(np.argmax(output))
            conf = float(output[cls_idx])
            mat_name = MATERIAL_CLASSES[cls_idx] if cls_idx < len(MATERIAL_CLASSES) else "unknown"
            return mat_name, conf

        return "unknown_material", 0.0

    def refine_detection(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
    ) -> DetectionResult:
        """
        Crop detection, predict material, and compute fused confidence score.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = clip_bbox(detection.bbox, w, h)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return detection

        material_pred, mat_conf = self.predict_material(crop)
        if not self.is_ready:
            return detection

        detection.material = material_pred

        # Fused Confidence:
        # Check if material is compatible with YOLO class
        compatible_materials = MATERIAL_COMPATIBILITY.get(detection.class_name, [])
        if compatible_materials and material_pred in compatible_materials:
            # Boost confidence for agreement
            boost = 1.05
            fused_conf = min(1.0, ((1 - self.fusion_weight) * detection.confidence + self.fusion_weight * mat_conf) * boost)
        else:
            # Slight penalty for conflicting material signal
            fused_conf = (1 - self.fusion_weight) * detection.confidence + self.fusion_weight * mat_conf

        detection.confidence = float(fused_conf)
        return detection
