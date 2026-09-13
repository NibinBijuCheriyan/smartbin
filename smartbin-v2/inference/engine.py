"""
Multi-Backend Edge Inference Engine for SmartBin AI v2.
Supports PyTorch (.pt), ONNX Runtime (.onnx), OpenVINO IR (.xml), and TFLite (.tflite).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import cv2
import numpy as np

from smartbin_v2.utils.logger import get_logger

logger = get_logger("inference_engine")


@dataclass
class DetectionResult:
    """Standardized detection entity across all inference backends."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel coordinates
    track_id: int = -1
    mask: Optional[np.ndarray] = None  # Binary segmentation mask if available
    pixel_area: float = 0.0
    material: Optional[str] = None
    is_contaminated: bool = False
    priority_score: float = 0.0


class BaseInferenceEngine(ABC):
    """Abstract interface for all model execution runtimes."""

    def __init__(
        self, model_path: str | Path, conf_threshold: float = 0.45,
        iou_threshold: float = 0.50, class_names: Optional[Mapping[int, str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = {int(class_id): name for class_id, name in (class_names or {}).items()}

    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[DetectionResult]:
        """Execute inference on a single BGR OpenCV image."""
        pass


class PyTorchEngine(BaseInferenceEngine):
    """Ultralytics PyTorch inference backend."""

    def __init__(self, model_path: str | Path, conf_threshold: float = 0.45, iou_threshold: float = 0.50, device: str = "auto", class_names: Optional[Mapping[int, str]] = None) -> None:
        super().__init__(model_path, conf_threshold, iou_threshold, class_names)
        import torch
        from ultralytics import YOLO
        logger.info(f"Loading PyTorch YOLO model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def infer(self, frame: np.ndarray) -> List[DetectionResult]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        detections: List[DetectionResult] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names

        # Masks if segmentation model
        masks = r.masks.data.cpu().numpy() if r.masks is not None else None

        for idx in range(len(boxes)):
            cls_id = classes[idx]
            cls_name = names.get(cls_id, str(cls_id))
            conf = float(confs[idx])
            box = tuple(boxes[idx].tolist())  # (x1, y1, x2, y2)
            
            mask_arr = masks[idx] if masks is not None else None
            pixel_area = float(np.sum(mask_arr > 0.5)) if mask_arr is not None else float((box[2]-box[0])*(box[3]-box[1]))

            detections.append(
                DetectionResult(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=box,
                    mask=mask_arr,
                    pixel_area=pixel_area,
                )
            )

        return detections


class ONNXEngine(BaseInferenceEngine):
    """ONNX Runtime edge execution backend with CPU / OpenVINO / TensorRT provider support."""

    def __init__(self, model_path: str | Path, conf_threshold: float = 0.45, iou_threshold: float = 0.50, class_names: Optional[Mapping[int, str]] = None, device: str = "auto") -> None:
        super().__init__(model_path, conf_threshold, iou_threshold, class_names)
        import onnxruntime as ort
        logger.info(f"Initializing ONNX Runtime engine: {self.model_path}")
        
        # Optimize execution providers for Raspberry Pi ARM64 or x86
        available_providers = ort.get_available_providers()
        selected_providers = []
        if "TensorrtExecutionProvider" in available_providers:
            selected_providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available_providers:
            selected_providers.append("CUDAExecutionProvider")
        selected_providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(str(self.model_path), sess_options, providers=selected_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 640, 640]
        self.target_w = self.input_shape[3] if len(self.input_shape) == 4 else 640
        self.target_h = self.input_shape[2] if len(self.input_shape) == 4 else 640

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = img.shape[:2]
        r = min(self.target_h / h, self.target_w / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = self.target_w - new_unpad[0], self.target_h - new_unpad[1]
        dw /= 2
        dh /= 2

        if (w, h) != new_unpad:
            resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = img

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        return blob, r, (dw, dh)

    def infer(self, frame: np.ndarray) -> List[DetectionResult]:
        blob, ratio, (dw, dh) = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        # Parse standard YOLO output [1, 4 + nc, N]
        pred = outputs[0]
        if len(pred.shape) == 3 and pred.shape[1] < pred.shape[2]:
            pred = pred.transpose(0, 2, 1)

        boxes_list = []
        confidences = []
        class_ids = []

        for row in pred[0]:
            cx, cy, w, h = row[:4]
            scores = row[4:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])

            if conf > self.conf_threshold:
                # Rescale to original frame coords
                x1 = (cx - w / 2.0 - dw) / ratio
                y1 = (cy - h / 2.0 - dh) / ratio
                x2 = (cx + w / 2.0 - dw) / ratio
                y2 = (cy + h / 2.0 - dh) / ratio

                x1 = max(0.0, min(x1, frame.shape[1]))
                y1 = max(0.0, min(y1, frame.shape[0]))
                x2 = max(0.0, min(x2, frame.shape[1]))
                y2 = max(0.0, min(y2, frame.shape[0]))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes_list.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(conf)
                class_ids.append(cls_id)

        indices = cv2.dnn.NMSBoxes(boxes_list, confidences, self.conf_threshold, self.iou_threshold)
        detections: List[DetectionResult] = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                bx, by, bw, bh = boxes_list[i]
                detections.append(
                    DetectionResult(
                        class_id=class_ids[i],
                        class_name=self.class_names.get(class_ids[i], f"unknown_class_{class_ids[i]}"),
                        confidence=confidences[i],
                        bbox=(float(bx), float(by), float(bx + bw), float(by + bh)),
                        pixel_area=float(bw * bh),
                    )
                )

        return detections


def create_inference_engine(backend: str, model_path: str | Path, **kwargs: Any) -> BaseInferenceEngine:
    """Factory creating the appropriate inference engine."""
    backend = backend.lower()
    path = Path(model_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configured model artifact does not exist: {path}")
    if backend == "pytorch" or path.suffix == ".pt":
        return PyTorchEngine(model_path, **kwargs)
    elif backend == "onnx" or path.suffix == ".onnx":
        return ONNXEngine(model_path, **kwargs)
    raise ValueError(f"Unsupported inference backend '{backend}' for {path.suffix} model")
