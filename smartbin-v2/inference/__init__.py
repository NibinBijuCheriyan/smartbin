"""
SmartBin AI v2 - Production Inference Engine
"""
from smartbin_v2.inference.engine import BaseInferenceEngine, create_inference_engine, DetectionResult
from smartbin_v2.inference.tracker import ByteTrackerWrapper
from smartbin_v2.inference.prioritizer import MultiObjectPrioritizer
from smartbin_v2.inference.temporal_voter import TemporalVotingBuffer
from smartbin_v2.inference.material_classifier import MaterialClassifier
from smartbin_v2.inference.contamination_detector import ContaminationDetector
from smartbin_v2.inference.pipeline import SmartBinInferencePipeline

__all__ = [
    "BaseInferenceEngine",
    "create_inference_engine",
    "DetectionResult",
    "ByteTrackerWrapper",
    "MultiObjectPrioritizer",
    "TemporalVotingBuffer",
    "MaterialClassifier",
    "ContaminationDetector",
    "SmartBinInferencePipeline",
]
