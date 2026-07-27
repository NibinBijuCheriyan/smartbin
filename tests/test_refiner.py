"""
Unit tests for the second-stage WasteClassifier refiner module.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smartbin.config import RefinerConfig
from smartbin.detector import Detection
from smartbin.refiner import (
    UNSUPPORTED_REFINER_CLASSES,
    WasteClassifier,
    create_refiner,
)


def test_detection_with_refinement():
    """Test Detection.with_refinement creates updated detection with raw YOLO metadata."""
    det = Detection(
        track_id=1,
        class_id=0,
        class_name="plastic",
        confidence=0.8,
        bbox=(10, 10, 100, 100),
        raw_yolo_class="plastic",
        raw_yolo_conf=0.8,
    )
    refined = det.with_refinement("paper", 0.95)

    assert refined.class_name == "paper"
    assert refined.confidence == 0.95
    assert refined.raw_yolo_class == "plastic"
    assert refined.raw_yolo_conf == 0.8
    assert refined.is_refined is True


def test_refiner_preprocessing():
    """Test crop preprocessing converts BGR crop to (1, 224, 224, 3) float32 array."""
    classifier = WasteClassifier(
        model_path="dummy.tflite",
        classes_path="dummy.json",
    )
    crop = np.zeros((100, 150, 3), dtype=np.uint8)
    processed = classifier.preprocess(crop)

    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == np.float32


def test_refiner_classify_mock_interpreter():
    """Test classify method using a mocked TFLite interpreter."""
    mock_interpreter = MagicMock()
    mock_interpreter.get_input_details.return_value = [{"index": 0, "dtype": np.float32, "shape": [1, 224, 224, 3]}]
    mock_interpreter.get_output_details.return_value = [{"index": 1}]
    # 5 classes: plastic, paper, metal, organic_waste, none. Index 1 = paper
    mock_interpreter.get_tensor.return_value = np.array([[0.05, 0.85, 0.05, 0.03, 0.02]], dtype=np.float32)

    classifier = WasteClassifier(
        model_path="dummy.tflite",
        classes_path="cashcrow-classification-model/efficientnet_b0_224_5class_int8/classes.json",
        interpreter=mock_interpreter,
    )

    crop = np.ones((50, 50, 3), dtype=np.uint8) * 128
    predicted_class, confidence = classifier.classify(crop)

    assert predicted_class == "paper"
    assert pytest.approx(confidence, 0.01) == 0.85
    mock_interpreter.invoke.assert_called_once()


def test_unsupported_classes_skipped():
    """Verify glass and e-waste are in UNSUPPORTED_REFINER_CLASSES."""
    assert "glass" in UNSUPPORTED_REFINER_CLASSES
    assert "e-waste" in UNSUPPORTED_REFINER_CLASSES


def test_create_refiner_disabled():
    """Test create_refiner returns None when refiner config disabled."""
    cfg = RefinerConfig(enabled=False)
    refiner = create_refiner(cfg)
    assert refiner is None
