"""
Cashcrow Smartbin — CLI entry point.

Usage:
    # Run with default config against webcam
    python main.py

    # Run against a video file with live preview
    python main.py --source path/to/video.mp4 --show

    # Use custom model weights and confidence threshold
    python main.py --weights best.pt --confidence 0.5

    # Use a custom config file
    python main.py --config my_config.yaml
"""

from __future__ import annotations

import sys
import logging

from smartbin.config import load_config, setup_logging
from smartbin.pipeline import SmartbinPipeline

logger = logging.getLogger("smartbin")


def main() -> None:
    """CLI entry point for the Smartbin pipeline."""
    try:
        # Load config (YAML + CLI overrides)
        config = load_config()

        # Set up logging
        setup_logging(config.logging)

        logger.info("=" * 60)
        logger.info("Cashcrow Smartbin — Starting pipeline")
        logger.info("=" * 60)
        logger.info("Model weights: %s", config.model.weights)
        logger.info("Video source:  %s", config.camera.source)
        logger.info("Confidence:    %.2f", config.model.confidence_threshold)
        logger.info("Hand tracking: %s (ROI crop: %s)", config.hand_tracking.enabled, config.hand_tracking.roi_crop_enabled)
        logger.info("Window size:   %d frames", config.buffer.active_window_size)
        logger.info("Idle timeout:  %d frames", config.buffer.idle_timeout_frames)
        logger.info("=" * 60)

        # Build and run the pipeline
        pipeline = SmartbinPipeline(config)
        pipeline.run()

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("Runtime error: %s", e)
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error — shutting down")
        sys.exit(1)


if __name__ == "__main__":
    main()
