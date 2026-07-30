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

    # Validate model weights without starting the camera loop
    python main.py --dry-run

    # Force run with generic (COCO) weights (not recommended)
    python main.py --allow-generic-model
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Set

from smartbin.config import load_config, setup_logging
from smartbin.pipeline import SmartbinPipeline

logger = logging.getLogger("smartbin")

# The 7 waste classes the Smartbin pipeline expects.
EXPECTED_WASTE_CLASSES: Set[str] = {
    "plastic", "paper", "metal", "glass", "e-waste", "organic", "other"
}


def validate_model_classes(
    weights_path: str,
    allow_generic: bool = False,
) -> bool:
    """
    Check whether the configured model weights contain the expected 7 waste
    classes. Returns True if validation passes, False otherwise.

    If the model contains generic COCO classes instead of waste classes, this
    indicates the model has not been fine-tuned for waste detection and will
    produce meaningless results.

    Args:
        weights_path: Path to the YOLO model weights.
        allow_generic: If True, log a warning but allow running anyway.

    Returns:
        True if the model is valid for waste detection (or allow_generic is set).
    """
    weights = Path(weights_path)

    # Skip validation for non-existent weights (will fail at model load time)
    if not weights.exists() and not weights_path.startswith("yolo"):
        return True

    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        model_classes = set(model.names.values()) if model.names else set()
    except Exception as e:
        logger.warning("Could not load model for class validation: %s", e)
        return True  # Don't block on validation errors

    if not model_classes:
        logger.warning("Model has no class names — cannot validate.")
        return True

    # Check if the model contains our expected waste classes
    waste_classes_present = EXPECTED_WASTE_CLASSES & model_classes
    is_waste_model = len(waste_classes_present) >= 5  # Allow minor variations

    if is_waste_model:
        logger.info(
            "Model class validation PASSED: %d/%d waste classes found.",
            len(waste_classes_present),
            len(EXPECTED_WASTE_CLASSES),
        )
        return True

    # Model appears to be generic (COCO or similar)
    logger.warning("=" * 70)
    logger.warning("MODEL CLASS MISMATCH DETECTED")
    logger.warning("=" * 70)
    logger.warning(
        "The configured model (%s) does not contain Smartbin waste classes.",
        weights_path,
    )
    logger.warning(
        "Found classes: %s",
        ", ".join(sorted(list(model_classes)[:10])) + ("..." if len(model_classes) > 10 else ""),
    )
    logger.warning(
        "Expected classes: %s",
        ", ".join(sorted(EXPECTED_WASTE_CLASSES)),
    )
    logger.warning("")
    logger.warning(
        "Running with generic (e.g. COCO) weights will NOT classify waste"
    )
    logger.warning(
        "correctly. Train a custom model first:"
    )
    logger.warning("  python train_waste_model.py")
    logger.warning("=" * 70)

    if allow_generic:
        logger.warning(
            "Proceeding with generic model (--allow-generic-model flag)."
        )
        logger.warning(
            "TIP: Consider passing --class-agnostic so detections bypass class filtering "
            "and reach the EfficientNet refiner."
        )
        return True

    logger.error(
        "Refusing to run with generic model weights. Use one of:"
    )
    logger.error("  --allow-generic-model  to run anyway (not recommended)")
    logger.error("  --weights best.pt      to use fine-tuned weights")
    logger.error("  python train_waste_model.py   to train a waste model")
    return False


def _apply_weight_fallbacks(config):
    """
    Handle missing weights and auto-enable class-agnostic mode.

    If the configured weights file (e.g. best.pt) does not exist, automatically
    fall back to yolo11n.pt with class-agnostic + allow-generic-model enabled.
    If allow_generic_model is already set with a non-waste model, auto-enable
    class_agnostic so detections aren't silently dropped.

    Returns a (possibly rebuilt) SmartbinConfig.
    """
    from smartbin.config import ModelConfig, SmartbinConfig

    weights_path = Path(config.model.weights)
    needs_rebuild = False
    new_weights = config.model.weights
    new_allow_generic = config.allow_generic_model
    new_class_agnostic = config.model.class_agnostic

    # --- Fallback: configured weights file missing ---
    if not weights_path.exists() and not config.model.weights.startswith("yolo"):
        fallback = Path("yolo11n.pt")
        if fallback.exists():
            logger.warning("=" * 60)
            logger.warning(
                "Configured weights '%s' not found on disk.",
                config.model.weights,
            )
            logger.warning(
                "AUTO-FALLBACK: Using '%s' as class-agnostic object locator "
                "with EfficientNet refiner for waste classification.",
                fallback,
            )
            logger.warning("=" * 60)
            new_weights = str(fallback)
            new_allow_generic = True
            new_class_agnostic = True
            needs_rebuild = True
        else:
            logger.error(
                "Configured weights '%s' not found and no fallback "
                "model (yolo11n.pt) available. Cannot start pipeline.",
                config.model.weights,
            )
            sys.exit(1)

    # --- Auto class-agnostic: generic model without the flag ---
    if new_allow_generic and not new_class_agnostic:
        logger.info(
            "Generic model enabled — auto-activating class-agnostic mode "
            "so COCO detections reach the EfficientNet refiner."
        )
        new_class_agnostic = True
        needs_rebuild = True

    if not needs_rebuild:
        return config

    return SmartbinConfig(
        model=ModelConfig(
            weights=new_weights,
            confidence_threshold=config.model.confidence_threshold,
            device=config.model.device,
            allowed_classes=config.model.allowed_classes,
            min_box_area_fraction=config.model.min_box_area_fraction,
            max_box_area_fraction=config.model.max_box_area_fraction,
            min_box_aspect_ratio=config.model.min_box_aspect_ratio,
            max_box_aspect_ratio=config.model.max_box_aspect_ratio,
            class_agnostic=new_class_agnostic,
        ),
        trigger=config.trigger,
        buffer=config.buffer,
        tracker=config.tracker,
        voter=config.voter,
        camera=config.camera,
        logging=config.logging,
        display=config.display,
        hand_tracking=config.hand_tracking,
        refiner=config.refiner,
        webhook=config.webhook,
        dry_run=config.dry_run,
        allow_generic_model=new_allow_generic,
    )


def main() -> None:
    """CLI entry point for the Smartbin pipeline."""
    try:
        # Load config (YAML + CLI overrides)
        config = load_config()

        # Set up logging
        setup_logging(config.logging)

        # Apply weight fallbacks (missing best.pt → yolo11n.pt + class-agnostic)
        config = _apply_weight_fallbacks(config)

        logger.info("=" * 60)
        logger.info("Cashcrow Smartbin — Starting pipeline")
        logger.info("=" * 60)
        logger.info("Model weights:   %s", config.model.weights)
        logger.info("Class-agnostic:  %s", config.model.class_agnostic)
        logger.info("Video source:    %s", config.camera.source)
        logger.info("Confidence:      %.2f", config.model.confidence_threshold)
        logger.info("Hand tracking:   %s (ROI crop: %s)", config.hand_tracking.enabled, config.hand_tracking.roi_crop_enabled)
        logger.info("Refiner:         %s", "enabled" if config.refiner.enabled else "disabled")
        logger.info("Window size:     %d frames", config.buffer.active_window_size)
        logger.info("Idle timeout:    %d frames", config.buffer.idle_timeout_frames)
        logger.info("=" * 60)

        # Validate model classes
        if not validate_model_classes(
            config.model.weights,
            allow_generic=config.allow_generic_model,
        ):
            sys.exit(1)

        # Dry-run mode: validate config and model, then exit
        if config.dry_run:
            logger.info("")
            logger.info("DRY RUN — config and model validated successfully.")
            logger.info("Remove --dry-run to start the camera loop.")
            return

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
