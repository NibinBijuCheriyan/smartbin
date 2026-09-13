"""
Production Model Evaluator for SmartBin AI v2.
Computes mAP@50, mAP@50-95, Precision, Recall, F1, Latency breakdown, FPS, and Memory footprints.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import psutil
from ultralytics import YOLO

from smartbin_v2.utils.logger import get_logger, setup_logging

logger = get_logger("evaluator")


@dataclass
class EvaluationMetrics:
    precision: float
    recall: float
    f1_score: float
    map50: float
    map50_95: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_latency_ms: float
    fps: float
    memory_ram_mb: float
    gpu_memory_mb: float
    per_class_metrics: Dict[str, Dict[str, float]]


class WasteModelEvaluator:
    """
    Evaluates Ultralytics YOLO models on validation/test splits with industrial metric reporting.
    """

    def __init__(
        self,
        weights_path: str = "best.pt",
        data_yaml: str = "smartbin-v2/datasets/smartbin_dataset.yaml",
        device: str = "auto",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.data_yaml = Path(data_yaml)
        self.device = device

        if not self.weights_path.exists():
            logger.warning(f"Weights {weights_path} not found; falling back to yolo11s.pt")
            self.model = YOLO("yolo11s.pt")
        else:
            self.model = YOLO(str(self.weights_path))

    def evaluate(self, split: str = "val", imgsz: int = 640) -> EvaluationMetrics:
        """
        Run validation and gather complete metrics.
        """
        logger.info(f"Evaluating model on {split} split (imgsz={imgsz})...")

        t0 = time.time()
        val_results = self.model.val(
            data=str(self.data_yaml),
            split=split,
            imgsz=imgsz,
            device=self.device,
            plots=True,
            verbose=False,
            project="smartbin-v2/runs/eval",
            name="latest_eval",
            exist_ok=True,
        )

        box = val_results.box
        precision = float(box.mp) if hasattr(box, "mp") else 0.0
        recall = float(box.mr) if hasattr(box, "mr") else 0.0
        map50 = float(box.map50) if hasattr(box, "map50") else 0.0
        map50_95 = float(box.map) if hasattr(box, "map") else 0.0

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Extract timing benchmarks
        speed = val_results.speed or {}
        prep_ms = float(speed.get("preprocess", 0.0))
        infer_ms = float(speed.get("inference", 0.0))
        post_ms = float(speed.get("postprocess", 0.0))
        total_lat = prep_ms + infer_ms + post_ms
        fps = (1000.0 / total_lat) if total_lat > 0 else 0.0

        # Memory usage
        process = psutil.Process()
        ram_mb = process.memory_info().rss / (1024 * 1024)

        # Per class breakdown
        per_class: Dict[str, Dict[str, float]] = {}
        names = val_results.names or {}
        if hasattr(box, "p") and hasattr(box, "r") and hasattr(box, "maps"):
            for idx, name in names.items():
                p_cls = float(box.p[idx]) if idx < len(box.p) else precision
                r_cls = float(box.r[idx]) if idx < len(box.r) else recall
                m50_cls = float(box.maps[idx]) if idx < len(box.maps) else map50
                per_class[name] = {
                    "precision": round(p_cls, 3),
                    "recall": round(r_cls, 3),
                    "map50": round(m50_cls, 3),
                }

        metrics = EvaluationMetrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            map50=round(map50, 4),
            map50_95=round(map50_95, 4),
            preprocess_ms=round(prep_ms, 2),
            inference_ms=round(infer_ms, 2),
            postprocess_ms=round(post_ms, 2),
            total_latency_ms=round(total_lat, 2),
            fps=round(fps, 1),
            memory_ram_mb=round(ram_mb, 1),
            gpu_memory_mb=0.0,
            per_class_metrics=per_class,
        )

        logger.info("--- Evaluation Complete ---")
        logger.info(f"mAP@50: {metrics.map50 * 100:.1f}% | mAP@50-95: {metrics.map50_95 * 100:.1f}%")
        logger.info(f"Precision: {metrics.precision * 100:.1f}% | Recall: {metrics.recall * 100:.1f}% | F1: {metrics.f1_score:.3f}")
        logger.info(f"Latency: {metrics.total_latency_ms} ms ({metrics.fps} FPS) | RAM: {metrics.memory_ram_mb} MB")

        return metrics
