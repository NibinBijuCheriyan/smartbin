"""
Master Inference Pipeline Orchestrator for SmartBin AI v2.
Wires: Camera Frame -> YOLO Detector -> ByteTracker -> Multi-Object Prioritizer ->
Material Classifier -> Contamination Check -> Temporal Voting Buffer -> Bin Decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from smartbin_v2.inference.contamination_detector import ContaminationDetector
from smartbin_v2.inference.engine import DetectionResult, create_inference_engine
from smartbin_v2.inference.material_classifier import MaterialClassifier
from smartbin_v2.inference.prioritizer import MultiObjectPrioritizer
from smartbin_v2.inference.temporal_voter import TemporalVotingBuffer, VotingDecision
from smartbin_v2.inference.tracker import ByteTrackerWrapper
from smartbin_v2.utils.active_learning import ActiveLearner
from smartbin_v2.utils.config import load_yaml_config
from smartbin_v2.utils.logger import get_logger

logger = get_logger("inference_pipeline")


@dataclass
class SegregationDecision:
    """Final actuation command ready for hardware servo dispatch."""
    target_bin: str  # "recyclable", "compost", "landfill", "reject"
    compartment_id: int  # 1, 2, 3, or 4
    class_name: str
    class_id: int
    confidence: float
    material: Optional[str]
    is_contaminated: bool
    track_id: int
    is_actuated: bool
    summary_message: str


class SmartBinInferencePipeline:
    """
    End-to-end computer vision segregation pipeline for SmartBin edge hardware.
    """

    def __init__(
        self,
        deployment_config_path: str = "smartbin-v2/configs/deployment_config.yaml",
        model_config_path: str = "smartbin-v2/configs/model_config.yaml",
    ) -> None:
        self.dep_cfg = load_yaml_config(deployment_config_path)
        self.model_cfg = load_yaml_config(model_config_path)

        # 1. Detection Engine
        rt = self.dep_cfg.get("runtime", {})
        model_settings = self.model_cfg.get("model", {})
        backend = rt.get("inference_backend", "onnx")
        model_path = rt.get("model_path", "smartbin-v2/models/smartbin_yolo11s.onnx")
        class_names = {int(class_id): name for class_id, name in self.model_cfg.get("class_names", {}).items()}
        expected_classes = self.model_cfg.get("taxonomy", {}).get("num_classes", 0)
        if len(class_names) != expected_classes:
            raise ValueError("Taxonomy class count does not match class_names mapping")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Configured production model is missing: {model_path}")

        self.engine = create_inference_engine(
            backend=backend,
            model_path=model_path,
            conf_threshold=model_settings.get("confidence_threshold", 0.45),
            iou_threshold=model_settings.get("iou_threshold", 0.50),
            device=model_settings.get("device", "auto"),
            class_names=class_names,
        )

        # 2. ByteTracker
        tr_cfg = self.dep_cfg.get("tracker", {})
        self.tracker = ByteTrackerWrapper(
            track_thresh=tr_cfg.get("track_thresh", 0.45),
            match_thresh=tr_cfg.get("match_thresh", 0.70),
            max_time_lost=tr_cfg.get("track_buffer", 25),
        )

        # 3. Multi-Object Prioritizer
        p_cfg = self.dep_cfg.get("prioritization", {}).get("weights", {})
        self.prioritizer = MultiObjectPrioritizer(
            weight_area=p_cfg.get("area", 0.40),
            weight_center_dist=p_cfg.get("center_distance", 0.40),
            min_area_fraction=self.dep_cfg.get("prioritization", {}).get("min_area_fraction", 0.005),
            max_area_fraction=self.dep_cfg.get("prioritization", {}).get("max_area_fraction", 0.85),
            weight_conf=p_cfg.get("confidence", 0.20),
        )

        # 4. Second-Stage Material Classifier
        mat_cfg = self.dep_cfg.get("material_classifier", {})
        self.material_classifier = MaterialClassifier(
            model_path=mat_cfg.get("model_path") if mat_cfg.get("enabled", True) else None,
            fusion_weight=mat_cfg.get("fusion_weight", 0.35),
        )

        # 5. Contamination Detector
        cont_cfg = self.dep_cfg.get("contamination_detector", {})
        self.contamination_detector = ContaminationDetector(
            model_path=cont_cfg.get("model_path") if cont_cfg.get("enabled", True) else None,
            threshold=cont_cfg.get("contamination_threshold", 0.65),
            reroute_to_landfill=cont_cfg.get("reroute_to_landfill", True),
        )

        # 6. Temporal Voting Buffer
        vote_cfg = self.dep_cfg.get("temporal_voting", {})
        self.voting_buffer = TemporalVotingBuffer(
            window_size=vote_cfg.get("window_size", 5),
            min_consensus_ratio=vote_cfg.get("min_consensus_ratio", 0.60),
        )

        # 7. Active Learning Module
        al_cfg = self.dep_cfg.get("active_learning", {})
        self.active_learner = ActiveLearner(
            queue_dir=al_cfg.get("queue_dir", "smartbin-v2/retraining_queue"),
            uncertainty_threshold=al_cfg.get("uncertainty_threshold", 0.55),
        ) if al_cfg.get("enabled", True) else None

        # 8. Bin Compartment Mappings
        self.compartment_mapping = self.model_cfg.get("compartment_mapping", {
            "recyclable": 1,
            "compost": 2,
            "landfill": 3,
            "reject": 4,
        })
        self._build_class_to_bin_lookup()

    def _build_class_to_bin_lookup(self) -> None:
        """Map class name to target bin from taxonomy."""
        self.class_to_bin: Dict[str, str] = {}
        tax = self.model_cfg.get("taxonomy", {}).get("categories", {})
        for cat_key, cat_data in tax.items():
            target_bin = cat_data.get("bin", "landfill")
            for cls_name in cat_data.get("classes", []):
                self.class_to_bin[cls_name] = target_bin

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[Optional[SegregationDecision], List[DetectionResult]]:
        """
        Process a single incoming camera frame through the full multi-stage pipeline.
        Returns: (final_actuation_decision_if_stable, all_current_frame_detections)
        """
        h, w = frame.shape[:2]

        # Stage 1: Object Detection
        detections = self.engine.infer(frame)

        # Stage 2: Temporal Tracking (ByteTrack)
        tracked_detections = self.tracker.update(detections)
        tracked_detections = [det for det in tracked_detections if det.class_name in self.class_to_bin]

        # Stage 3: Multi-Object Prioritization
        primary_item, ranked_detections = self.prioritizer.prioritize(
            tracked_detections, frame_width=w, frame_height=h
        )

        # If an item is dominant, refine with 2nd-stage heads
        if primary_item is not None:
            # Stage 4: Material Classification & Fusion
            primary_item = self.material_classifier.refine_detection(frame, primary_item)

            # Stage 5: Contamination Check
            primary_item = self.contamination_detector.evaluate_detection(frame, primary_item)

            # Stage 6: Push into 5-frame Temporal Voting Buffer
            self.voting_buffer.add_observation(primary_item)

            # Check for Active Learning enqueue
            if self.active_learner:
                self.active_learner.enqueue_sample(
                    frame=frame,
                    detected_classes=[primary_item.class_name],
                    confidences=[primary_item.confidence],
                )
        else:
            self.voting_buffer.add_observation(None)

        # Stage 7: Evaluate Window Consensus
        vote_result: Optional[VotingDecision] = self.voting_buffer.evaluate_consensus()

        if vote_result is None or not vote_result.is_stable:
            return None, ranked_detections

        # Stage 8: Determine Bin Compartment & Actuation Routing
        class_name = vote_result.confirmed_class
        base_bin = self.class_to_bin.get(class_name, "reject")

        # Routing logic:
        # If recyclable but contaminated -> Reroute to Landfill / Reject
        final_bin = base_bin
        reference = vote_result.reference_detection
        is_dirty = reference.is_contaminated
        if is_dirty and base_bin == "recyclable":
            final_bin = "landfill"
            summary = f"Contaminated {class_name} diverted from Recyclable to Landfill"
        elif vote_result.mean_confidence < 0.40:
            final_bin = "reject"
            summary = f"Unrecognized object ({class_name}, conf {vote_result.mean_confidence:.2f}) routed to Reject"
        else:
            summary = f"Item {class_name} segregated into {final_bin.upper()} bin"

        compartment_id = self.compartment_mapping.get(final_bin, 3)

        decision = SegregationDecision(
            target_bin=final_bin,
            compartment_id=compartment_id,
            class_name=class_name,
            class_id=vote_result.class_id,
            confidence=vote_result.mean_confidence,
            material=reference.material,
            is_contaminated=is_dirty,
            track_id=vote_result.track_id,
            is_actuated=True,
            summary_message=summary,
        )

        logger.info(f"FINAL DECISION: {summary}")
        # Reset buffer after confirmed decision to prevent duplicate triggers
        self.voting_buffer.reset()

        return decision, ranked_detections
