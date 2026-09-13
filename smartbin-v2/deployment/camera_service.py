"""
Threaded Camera Capture Service for SmartBin AI v2.
Runs background frame acquisition to decouple sensor read latency from AI inference
and eliminates video buffer delay.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from smartbin_v2.raspberry_pi.rpi_camera import RPiCameraDriver
from smartbin_v2.utils.logger import get_logger

logger = get_logger("camera_service")


class ThreadedCameraService:
    """
    Decoupled threaded camera worker continuously polling frames.
    Always provides the freshest frame to the inference pipeline without lag.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 640,
        fps: int = 30,
    ) -> None:
        self.driver = RPiCameraDriver(width=width, height=height, fps=fps, source=source)
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start acquisition background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraWorker")
        self._thread.start()
        logger.info("Threaded camera capture service started.")

    def _capture_loop(self) -> None:
        while self._running:
            success, frame = self.driver.read_frame()
            if success and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._frame_count += 1
            else:
                time.sleep(0.01)

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieve most recently captured frame."""
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
            return False, None

    def stop(self) -> None:
        """Stop worker and release hardware."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self.driver.release()
        logger.info("Threaded camera service stopped.")
