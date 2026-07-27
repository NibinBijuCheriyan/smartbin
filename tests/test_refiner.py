"""
Unit tests for the second-stage WasteClassifier refiner module.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from smartbin.config import RefinerConfig
from smartbin.detector import Detection
from smartbin.refiner import (
    UNSUPPORTED_REFINER_CLASSES,
    WasteClassifier,
    create_refiner,
)


def test_detection_with_refinement_syncs_class_id():
    """Test Detection.with_refinement updates class_id along with class_name and confidence."""
    det = Detection(
        track_id=1,
        class_id=0,
        class_name="plastic",
        confidence=0.8,
        bbox=(10, 10, 100, 100),
        raw_yolo_class="plastic",
        raw_yolo_conf=0.8,
    )
    refined = det.with_refinement(new_class_name="paper", new_confidence=0.95, new_class_id=1)

    assert refined.class_name == "paper"
    assert refined.confidence == 0.95
    assert refined.class_id == 1
    assert refined.raw_yolo_class == "plastic"
    assert refined.raw_yolo_conf == 0.8
    assert refined.is_refined is True


def test_eager_loading_fails_when_files_missing():
    """Fix #1: Test that create_refiner raises FileNotFoundError eagerly if files are missing."""
    cfg = RefinerConfig(
        enabled=True,
        model_path="non_existent_model.tflite",
        classes_path="non_existent_classes.json",
    )
    with pytest.raises(FileNotFoundError):
        create_refiner(cfg)


def test_confidence_threshold_enforced_returns_none_sentinel():
    """Fix #2: Test that sub-threshold prediction returns ('none', conf, -1)."""
    mock_interpreter = MagicMock()
    mock_interpreter.get_input_details.return_value = [{"index": 0, "dtype": np.float32, "shape": [1, 224, 224, 3]}]
    mock_interpreter.get_output_details.return_value = [{"index": 1}]
    # Highest confidence is 0.20 for paper (index 1), below threshold 0.25
    mock_interpreter.get_tensor.return_value = np.array([[0.15, 0.20, 0.15, 0.15, 0.15]], dtype=np.float32)

    classes_path = Path("cashcrow-classification-model/efficientnet_b0_224_5class_int8/classes.json")
    classifier = WasteClassifier(
        model_path="dummy.tflite",
        classes_path=classes_path if classes_path.exists() else "dummy.json",
        confidence_threshold=0.25,
        interpreter=mock_interpreter,
    )
    if not classes_path.exists():
        classifier._idx_to_class = {"0": "plastic", "1": "paper", "2": "metal", "3": "organic_waste", "4": "none"}
        classifier._class_to_idx = {"plastic": 0, "paper": 1, "metal": 2, "organic_waste": 3, "none": 4}

    crop = np.ones((50, 50, 3), dtype=np.uint8) * 128
    predicted_class, confidence, class_id = classifier.classify(crop)

    assert predicted_class == "none"
    assert pytest.approx(confidence, 0.01) == 0.20
    assert class_id == -1


def test_preprocessing_parity_with_predict_py():
    """Fix #5: Test that WasteClassifier.preprocess produces exact array parity with predict.py logic."""
    classes_path = Path("cashcrow-classification-model/efficientnet_b0_224_5class_int8/classes.json")
    classifier = WasteClassifier(
        model_path="dummy.tflite",
        classes_path=classes_path if classes_path.exists() else "dummy.json",
    )
    if not classes_path.exists():
        classifier._idx_to_class = {"0": "plastic", "1": "paper", "2": "metal", "3": "organic_waste", "4": "none"}

    # Generate a realistic test image crop (BGR 100x120x3)
    np.random.seed(42)
    bgr_crop = np.random.randint(0, 256, (100, 120, 3), dtype=np.uint8)

    # Reference predict.py preprocessing implementation
    import cv2
    rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    img_ref = Image.fromarray(rgb_crop).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    expected_array = np.array(img_ref, dtype=np.float32)
    expected_array = np.expand_dims(expected_array, axis=0)

    # Classifier preprocess output
    classifier_array = classifier.preprocess(bgr_crop)

    # Assert exact numerical array equality
    np.testing.assert_array_equal(classifier_array, expected_array)


def test_unsupported_classes_skipped():
    """Verify glass and e-waste are in UNSUPPORTED_REFINER_CLASSES."""
    assert "glass" in UNSUPPORTED_REFINER_CLASSES
    assert "e-waste" in UNSUPPORTED_REFINER_CLASSES


def test_create_refiner_disabled():
    """Test create_refiner returns None when refiner config disabled."""
    cfg = RefinerConfig(enabled=False)
    refiner = create_refiner(cfg)
    assert refiner is None
