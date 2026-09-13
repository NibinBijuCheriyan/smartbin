"""
Training Resume Utility for SmartBin AI v2.
Recovers training states seamlessly from last.pt checkpoints upon power loss or preemptions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO

from smartbin_v2.utils.logger import get_logger, setup_logging

logger = get_logger("training_resume")


def resume_training(checkpoint_path: str = "smartbin-v2/runs/train/smartbin_yolo11s/weights/last.pt") -> None:
    """Resume interrupted training job."""
    setup_logging(level="INFO")
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt}")

    logger.info(f"Resuming YOLO11 training from checkpoint: {ckpt}")
    model = YOLO(str(ckpt))
    model.train(resume=True)
    logger.info("Resumed training finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume YOLO11 Training")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="smartbin-v2/runs/train/smartbin_yolo11s/weights/last.pt",
        help="Path to checkpoint file (last.pt)",
    )
    args = parser.parse_args()
    resume_training(args.checkpoint)
