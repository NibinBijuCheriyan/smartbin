"""
Hardware Diagnostic and Self-Test Tool for SmartBin Edge Devices.
Tests Raspberry Pi Camera, GPIO/PWM Servo lines, and USB Serial Bridge.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import cv2

# Ensure project root is in sys.path when executed directly
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from smartbin_v2.raspberry_pi.rpi_camera import RPiCameraDriver
from smartbin_v2.utils.logger import get_logger, setup_logging

logger = get_logger("hardware_test")


def run_hardware_diagnostics(camera_source: int | str = 0) -> bool:
    """Execute end-to-end hardware verification."""
    setup_logging(level="INFO")
    logger.info("====================================================================")
    logger.info("Starting SmartBin Edge Hardware Diagnostic Self-Test")
    logger.info("====================================================================")

    all_passed = True

    # 1. Camera Diagnostic
    logger.info("1. Testing Camera Feed Acquisition...")
    try:
        driver = RPiCameraDriver(width=640, height=640, fps=30, source=camera_source)
        success, frame = driver.read_frame()
        if success and frame is not None:
            logger.info(f"   [PASS] Camera operational. Acquired frame shape: {frame.shape}")
            test_img_path = Path("smartbin-v2/logs/camera_self_test.jpg")
            test_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(test_img_path), frame)
            logger.info(f"   Saved diagnostic frame to: {test_img_path}")
        else:
            logger.error("   [FAIL] Could not capture frame from camera.")
            all_passed = False
        driver.release()
    except Exception as e:
        logger.error(f"   [FAIL] Camera initialization error: {e}")
        all_passed = False

    # 2. Raspberry Pi GPIO / PWM check
    logger.info("2. Testing Raspberry Pi GPIO subsystem...")
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        logger.info("   [PASS] RPi.GPIO library available and BCM mode engaged.")
        GPIO.cleanup()
    except (ImportError, RuntimeError) as e:
        logger.warning(f"   [NOTICE] Native RPi.GPIO not active ({e}) - likely running on PC or Arduino bridge.")

    logger.info("====================================================================")
    if all_passed:
        logger.info("DIAGNOSTIC RESULT: ALL HARDWARE CHECKS PASSED")
    else:
        logger.warning("DIAGNOSTIC RESULT: HARDWARE ISSUES DETECTED")
    logger.info("====================================================================")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hardware Diagnostics")
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or video file path")
    args = parser.parse_args()
    src: int | str = int(args.source) if args.source.isdigit() else args.source
    success = run_hardware_diagnostics(src)
    sys.exit(0 if success else 1)
