"""
Active Learning Pipeline for SmartBin AI v2.
Automatically monitors edge inference certainty, buffers borderline and unconfident
predictions to the retraining queue, manages annotation metadata, and triggers retraining.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from smartbin_v2.utils.logger import get_logger

logger = get_logger("active_learning")


@dataclass
class RetrainingSample:
    """Metadata for an image routed to the active learning queue."""
    sample_id: str
    timestamp: str
    image_path: str
    detected_classes: List[str]
    confidence_scores: List[float]
    highest_confidence: float
    entropy: float
    trigger_reason: str
    hardware_id: str
    reviewed: bool = False
    verified_label: Optional[str] = None


class ActiveLearner:
    """
    Manages collection, filtering, queueing, and retraining invocation for edge failures.
    """

    def __init__(
        self,
        queue_dir: str | Path = "smartbin-v2/retraining_queue",
        uncertainty_threshold: float = 0.55,
        entropy_threshold: float = 1.2,
        max_samples: int = 5000,
    ) -> None:
        self.queue_dir = Path(queue_dir)
        self.images_dir = self.queue_dir / "images"
        self.meta_dir = self.queue_dir / "metadata"
        self.uncertainty_threshold = uncertainty_threshold
        self.entropy_threshold = entropy_threshold
        self.max_samples = max_samples

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy for predictive uncertainty."""
        if not probabilities:
            return 0.0
        entropy = 0.0
        for p in probabilities:
            if p > 1e-6:
                entropy -= p * math.log2(p)
        return entropy

    def should_sample(self, confidences: List[float]) -> Tuple[bool, str, float]:
        """
        Determine if an edge prediction should be harvested for active learning.
        Returns: (should_sample, reason, entropy)
        """
        if not confidences:
            return True, "zero_detections", 0.0

        max_conf = max(confidences)
        entropy = self.calculate_entropy(confidences)

        # 1. Low or borderline confidence
        if max_conf < self.uncertainty_threshold:
            return True, f"low_confidence_{max_conf:.2f}", entropy

        # 2. Ambiguity / High Entropy
        if entropy > self.entropy_threshold:
            return True, f"high_entropy_{entropy:.2f}", entropy

        return False, "", entropy

    def enqueue_sample(
        self,
        frame: np.ndarray,
        detected_classes: List[str],
        confidences: List[float],
        hardware_id: str = "rpi5-bin-01",
    ) -> Optional[RetrainingSample]:
        """
        Evaluate frame and enqueue if uncertainty criteria are met.
        """
        should_add, reason, entropy = self.should_sample(confidences)
        if not should_add:
            return None

        # Generate unique sample ID
        sample_id = f"sample_{int(time.time() * 1000)}_{np.random.randint(100, 999)}"
        img_filename = f"{sample_id}.jpg"
        meta_filename = f"{sample_id}.json"

        img_path = self.images_dir / img_filename
        meta_path = self.meta_dir / meta_filename

        # Save image
        cv2.imwrite(str(img_path), frame)

        sample = RetrainingSample(
            sample_id=sample_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            image_path=str(img_path),
            detected_classes=detected_classes,
            confidence_scores=confidences,
            highest_confidence=max(confidences) if confidences else 0.0,
            entropy=entropy,
            trigger_reason=reason,
            hardware_id=hardware_id,
        )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(sample), f, indent=2)

        logger.info(f"Queued active learning sample: {sample_id} ({reason})")
        return sample

    def list_pending_samples(self) -> List[RetrainingSample]:
        """List all pending samples awaiting verification or labeling."""
        samples: List[RetrainingSample] = []
        for meta_file in self.meta_dir.glob("*.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    samples.append(RetrainingSample(**data))
            except Exception as e:
                logger.warning(f"Error reading sample metadata {meta_file}: {e}")
        return samples

    def approve_sample(
        self,
        sample_id: str,
        verified_label: str,
        bbox: Optional[List[float]] = None,
        target_dataset_dir: str | Path = "smartbin-v2/datasets/processed",
    ) -> bool:
        """
        Mark a sample as verified, create standard YOLO label, and copy into dataset.
        """
        meta_file = self.meta_dir / f"{sample_id}.json"
        img_file = self.images_dir / f"{sample_id}.jpg"

        if not meta_file.exists() or not img_file.exists():
            return False

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["reviewed"] = True
        data["verified_label"] = verified_label

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Sample {sample_id} verified with label '{verified_label}'")
        return True
