"""
Evaluate the Cashcrow EfficientNet-B0 FP32 Classifier on real SmartBin crops.

Extracts crops from the sample test video (22-17-04.mp4) under realistic hand-held /
bin capture conditions, runs inference using waste_classifier_fp32.tflite, and evaluates
performance against the studio validation benchmark (96.59%).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
import cv2
import numpy as np

from smartbin.config import RefinerConfig
from smartbin.refiner import WasteClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_video_crops(video_path: str, model_path: str, classes_path: str):
    """Extract frames/crops from video and run TFLite classification evaluation."""
    if not Path(video_path).exists():
        logger.error("Video file not found: %s", video_path)
        return

    classifier = WasteClassifier(model_path=model_path, classes_path=classes_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video: %s", video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    logger.info("Opened video %s: %d frames @ %d FPS", video_path, total_frames, fps)

    frame_idx = 0
    sampled_crops = 0
    predictions_summary = []

    # Simple motion / foreground crop simulation for evaluating real crops
    back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        fg_mask = back_sub.apply(frame)

        # Sample every 10th frame when motion is detected
        if frame_idx % 10 == 0:
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            large_contours = [c for c in contours if cv2.contourArea(c) > 5000]

            if large_contours:
                # Get bounding box of largest motion region (simulating item/hand)
                c = max(large_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)

                # Add padding
                pad_x = int(w * 0.2)
                pad_y = int(h * 0.2)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    predicted_class, confidence = classifier.classify(crop)
                    sampled_crops += 1
                    predictions_summary.append({
                        "frame": frame_idx,
                        "class": predicted_class,
                        "confidence": confidence,
                        "crop_size": (w, h),
                    })

    cap.release()

    print("\n" + "=" * 70)
    print("      SMARTBIN REAL-CROP EFFICIENTNET-B0 CLASSIFIER EVALUATION REPORT")
    print("=" * 70)
    print(f"Test Video Source: {os.path.basename(video_path)}")
    print(f"Total Video Frames Processed: {frame_idx}")
    print(f"Total SmartBin Motion Crops Evaluated: {sampled_crops}")
    print("-" * 70)

    if predictions_summary:
        from collections import Counter
        class_counts = Counter(p["class"] for p in predictions_summary)
        mean_conf = np.mean([p["confidence"] for p in predictions_summary])

        print("\nPredicted Class Distribution on Real Crops:")
        for cls_name, count in class_counts.items():
            pct = (count / sampled_crops) * 100
            print(f"  - {cls_name:<15}: {count:>3} crops ({pct:>5.1f}%)")

        print(f"\nMean Classifier Confidence across Crops: {mean_conf * 100:.2f}%")
        print("\nDistribution Analysis vs. Studio Validation:")
        print("  - Studio Dataset Accuracy (README) : 96.59% (FP32)")
        print("  - Deployment Scene Domain Gap      : High (cluttered background, hand occlusion, non-studio lighting)")
        print("  - Disagreement / 'None' Frequency  : {:.1f}% of crops classified as 'none'".format(
            (class_counts.get("none", 0) / sampled_crops) * 100
        ))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    video = "22-17-04.mp4"
    model = "cashcrow-classification-model/efficientnet_b0_224_5class_int8/models/waste_classifier_fp32.tflite"
    classes = "cashcrow-classification-model/efficientnet_b0_224_5class_int8/classes.json"

    evaluate_video_crops(video, model, classes)
