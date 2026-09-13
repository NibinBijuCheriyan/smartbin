"""Active learning flagging script.

Runs best.pt over a video and logs low-confidence or "other"-class detections
to a CSV for manual relabeling. Saves cropped bbox images to disk.

Usage:
    python active_learning_flag.py --video 21-09-03.mp4
    python active_learning_flag.py --video 22-17-04.mp4 --weights best.pt --output flagged.csv
"""

import argparse
import csv
import logging
import os
from pathlib import Path

import cv2

from smartbin.config import ModelConfig, TrackerConfig
from smartbin.detector import YOLODetector

logger = logging.getLogger(__name__)

# Detections below this confidence OR belonging to this class are flagged
FLAG_CONFIDENCE_THRESHOLD = 0.5
FLAG_CLASS = "other"

DEFAULT_VIDEO = "21-09-03.mp4"
DEFAULT_WEIGHTS = "best.pt"
DEFAULT_OUTPUT_CSV = "active_learning_flags.csv"
DEFAULT_CROP_DIR = "active_learning_crops"

CSV_COLUMNS = ["frame_number", "track_id", "predicted_class", "confidence", "crop_path"]


def should_flag(class_name: str, confidence: float) -> bool:
    """Return True if a detection should be flagged for manual review."""
    return confidence < FLAG_CONFIDENCE_THRESHOLD or class_name.lower() == FLAG_CLASS


def run_active_learning_flagging(
    video_path: str,
    weights: str,
    output_csv: str,
    crop_dir: str,
) -> int:
    """Run inference over a video and flag uncertain detections.

    Args:
        video_path: Path to input video file.
        weights: Path to YOLO model weights.
        output_csv: Path to output CSV file.
        crop_dir: Directory to save cropped bbox images.

    Returns:
        Number of detections flagged.
    """
    # Validate inputs
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not Path(weights).exists():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    # Create crop output directory
    Path(crop_dir).mkdir(parents=True, exist_ok=True)

    # Initialize detector with a low global threshold so we catch
    # uncertain detections that would normally be filtered out.
    model_config = ModelConfig(
        weights=weights,
        confidence_threshold=0.15,  # Very low — we want to catch borderline cases
        device="auto",
    )
    tracker_config = TrackerConfig()
    detector = YOLODetector(model_config, tracker_config)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    flagged_count = 0
    frame_num = 0

    try:
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
            writer.writeheader()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                detections = detector.detect(frame)

                for det in detections:
                    if not should_flag(det.class_name, det.confidence):
                        continue

                    # Save cropped bbox image
                    x1, y1, x2, y2 = map(int, det.bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    crop_filename = (
                        f"frame{frame_num:06d}_track{det.track_id}"
                        f"_{det.class_name}_{det.confidence:.3f}.jpg"
                    )
                    crop_path = os.path.join(crop_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)

                    writer.writerow({
                        "frame_number": frame_num,
                        "track_id": det.track_id,
                        "predicted_class": det.class_name,
                        "confidence": f"{det.confidence:.4f}",
                        "crop_path": crop_path,
                    })
                    flagged_count += 1

                if frame_num % 100 == 0:
                    logger.info(
                        "Processed %d frames, %d detections flagged so far",
                        frame_num, flagged_count,
                    )
    finally:
        cap.release()
        detector.close()

    logger.info(
        "Done. Processed %d frames, flagged %d detections → %s",
        frame_num, flagged_count, output_csv,
    )
    return flagged_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flag uncertain detections for active learning relabeling.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=DEFAULT_VIDEO,
        help=f"Path to input video (default: {DEFAULT_VIDEO}).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help=f"Path to YOLO model weights (default: {DEFAULT_WEIGHTS}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--crop-dir",
        type=str,
        default=DEFAULT_CROP_DIR,
        help=f"Directory for cropped bbox images (default: {DEFAULT_CROP_DIR}).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    flagged = run_active_learning_flagging(
        video_path=args.video,
        weights=args.weights,
        output_csv=args.output,
        crop_dir=args.crop_dir,
    )
    print(f"\nFlagged {flagged} detections. CSV: {args.output}, Crops: {args.crop_dir}")


if __name__ == "__main__":
    main()
