"""
Raspberry Pi Camera Module 3 Hardware Interface.
Supports libcamera / Picamera2 native acceleration on Raspberry Pi 5
with seamless fallback to OpenCV V4L2 on generic Linux / Windows.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path when executed directly
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from smartbin_v2.utils.logger import get_logger

logger = get_logger("rpi_camera")


class RPiCameraDriver:
    """
    High-performance camera acquisition driver for Raspberry Pi Camera Module 3.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 640,
        fps: int = 30,
        source: int | str = 0,
        prefer_picamera2: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.source = source
        self.prefer_picamera2 = prefer_picamera2

        self.picam2 = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.backend_type = "none"

        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """Initialize camera hardware."""
        if self.prefer_picamera2:
            try:
                from picamera2 import Picamera2
                logger.info("Initializing Raspberry Pi Camera Module 3 via native Picamera2...")
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    controls={"FrameRate": self.fps}
                )
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(1.0)  # Sensor warmup and auto-exposure lock
                self.backend_type = "picamera2"
                logger.info("Picamera2 initialized successfully.")
                return
            except Exception as e:
                logger.warning(f"Picamera2 not available ({e}). Falling back to OpenCV VideoCapture.")

        # OpenCV Fallback
        logger.info(f"Opening video source via OpenCV: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.backend_type = "opencv"

        if not self.cap.isOpened():
            logger.error(f"Failed to open video source {self.source}")
        else:
            logger.info("OpenCV VideoCapture initialized successfully.")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture latest frame from sensor in BGR format.
        """
        if self.backend_type == "picamera2" and self.picam2 is not None:
            try:
                rgb = self.picam2.capture_array()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return True, bgr
            except Exception as e:
                logger.error(f"Picamera2 capture failure: {e}")
                return False, None

        elif self.backend_type == "opencv" and self.cap is not None:
            ret, frame = self.cap.read()
            return ret, frame

        return False, None

    def release(self) -> None:
        """Release camera hardware resources."""
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()
        logger.info("Camera resources released.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RPiCameraDriver on Laptop or Raspberry Pi")
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or video file path")
    parser.add_argument("--preview", action="store_true", help="Display live camera preview window")
    args = parser.parse_args()

    src: int | str = int(args.source) if args.source.isdigit() else args.source
    print(f"--- Testing RPiCameraDriver (source={src!r}) ---")
    driver = RPiCameraDriver(width=640, height=480, fps=30, source=src)
    print(f"Active Backend: {driver.backend_type}")

    if args.preview:
        print("Opening live preview window. Press 'q' inside the window to exit.")
        while True:
            ret, frame = driver.read_frame()
            if not ret or frame is None:
                print("End of stream or failed to grab frame.")
                break
            cv2.imshow("RPiCameraDriver Laptop Test (Press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
    else:
        ret, frame = driver.read_frame()
        if ret and frame is not None:
            print(f"[PASS] Successfully captured frame! Dimensions: {frame.shape}, Dtype: {frame.dtype}")
        else:
            print(f"[FAIL] Could not capture frame from source {src!r}.")

    driver.release()
    print("Test completed.")
