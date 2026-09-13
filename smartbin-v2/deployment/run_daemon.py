"""
Production Daemon Entrypoint for SmartBin Edge Devices.
Starts threaded camera, inference pipeline, decision broker, and watchdog.
Handles graceful shutdowns on SIGINT / SIGTERM.
"""

from __future__ import annotations

import signal
import sys
import time
from typing import Any

from smartbin_v2.arduino.serial_bridge import ArduinoSerialBridge
from smartbin_v2.deployment.camera_service import ThreadedCameraService
from smartbin_v2.deployment.decision_engine import DecisionEngine
from smartbin_v2.deployment.health_monitor import HealthWatchdog
from smartbin_v2.deployment.offline_queue import OfflineEventQueue
from smartbin_v2.inference.pipeline import SmartBinInferencePipeline
from smartbin_v2.utils.logger import get_logger, setup_logging

logger = get_logger("daemon")


class SmartBinEdgeDaemon:
    """Industrial background daemon coordinating hardware and AI inference."""

    def __init__(self) -> None:
        self.running = True
        setup_logging(level="INFO")
        logger.info("Initializing SmartBin AI v2 Edge Daemon...")

        # Initialize hardware and services
        self.camera_service = ThreadedCameraService()
        self.arduino = ArduinoSerialBridge()
        self.offline_queue = OfflineEventQueue()
        self.decision_engine = DecisionEngine(
            arduino_bridge=self.arduino,
            offline_queue=self.offline_queue,
        )
        self.pipeline = SmartBinInferencePipeline()
        self.watchdog = HealthWatchdog()

        # Signal handlers for systemd shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        logger.info(f"Shutdown signal ({signum}) received. Stopping daemon...")
        self.running = False

    def run(self) -> None:
        """Main operational execution loop."""
        logger.info("Starting camera worker thread...")
        self.camera_service.start()
        logger.info("Daemon active. Processing frames...")

        last_health_check = time.time()

        try:
            while self.running:
                success, frame = self.camera_service.get_latest_frame()
                if not success or frame is None:
                    time.sleep(0.01)
                    continue

                # Notify watchdog of frame liveness
                self.watchdog.ping_camera_alive()

                # Execute multi-stage segregation pipeline
                decision, detections = self.pipeline.process_frame(frame)

                if decision is not None and decision.is_actuated:
                    self.decision_engine.dispatch_decision(decision)

                # Periodic Health Watchdog check (every 30 seconds)
                now = time.time()
                if now - last_health_check >= 30.0:
                    snapshot = self.watchdog.check_health()
                    logger.info(
                        f"Health: {snapshot.status} | Temp: {snapshot.temperature_celsius}°C "
                        f"| CPU: {snapshot.cpu_percent}% | RAM: {snapshot.memory_used_mb}MB"
                    )
                    last_health_check = now

                time.sleep(0.005)

        except Exception as e:
            logger.critical(f"Unhandled exception in daemon loop: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self) -> None:
        """Cleanup all hardware connections."""
        logger.info("Shutting down camera service...")
        self.camera_service.stop()
        logger.info("Disconnecting serial bridge...")
        self.arduino.close()
        logger.info("SmartBin AI v2 Daemon shutdown complete.")


if __name__ == "__main__":
    daemon = SmartBinEdgeDaemon()
    daemon.run()
