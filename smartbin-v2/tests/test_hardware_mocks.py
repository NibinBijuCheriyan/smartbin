"""
Hardware mock tests verifying Arduino serial bridge, offline queue, and watchdog.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from smartbin_v2.arduino.serial_bridge import ArduinoSerialBridge
from smartbin_v2.deployment.health_monitor import HealthWatchdog
from smartbin_v2.deployment.offline_queue import OfflineEventQueue
from smartbin_v2.raspberry_pi.rpi_camera import RPiCameraDriver


def test_arduino_mock_bridge():
    """Verify Arduino serial bridge in simulated mock mode."""
    bridge = ArduinoSerialBridge(mock_mode=True)
    assert bridge.is_connected is True

    # Test Heartbeat Ping
    assert bridge.ping() is True

    # Test Compartment Actuation
    assert bridge.trigger_sort(1) is True
    assert bridge.trigger_sort(2) is True


def test_health_watchdog():
    """Verify watchdog captures valid system telemetry."""
    watchdog = HealthWatchdog()
    watchdog.ping_camera_alive()

    snapshot = watchdog.check_health()
    assert snapshot.status in ["HEALTHY", "THERMAL_THROTTLING_WARNING", "HIGH_CPU_LOAD"]
    assert snapshot.cpu_percent >= 0.0
    assert snapshot.memory_used_mb > 0.0
    assert snapshot.camera_alive is True


def test_offline_event_queue(tmp_path: Path):
    """Verify SQLite persistent offline queue enqueue and retrieval."""
    test_db = tmp_path / "test_events.db"
    queue = OfflineEventQueue(db_path=test_db)

    test_event = {"class_name": "pet_bottle", "target_bin": "recyclable", "confidence": 0.95}
    ev_id = queue.enqueue(test_event)
    assert ev_id > 0

    pending = queue.get_pending(limit=10)
    assert len(pending) == 1
    assert pending[0][0] == ev_id
    assert pending[0][1]["class_name"] == "pet_bottle"

    queue.mark_synced(ev_id)
    assert len(queue.get_pending(limit=10)) == 0


def test_rpi_camera_opencv_fallback():
    """Verify fallback to OpenCV when Picamera2 is unavailable."""
    driver = RPiCameraDriver(source="test_video.avi", prefer_picamera2=False)
    assert driver.backend_type == "opencv"
    ret, frame = driver.read_frame()
    assert ret is True
    assert frame is not None
    assert len(frame.shape) == 3
    driver.release()


def test_rpi_camera_mock_picamera2():
    """Verify Picamera2 initialization and frame capture using mock."""
    mock_picam2_cls = MagicMock()
    mock_inst = mock_picam2_cls.return_value
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_inst.capture_array.return_value = dummy_rgb

    with patch.dict("sys.modules", {"picamera2": MagicMock(Picamera2=mock_picam2_cls)}):
        driver = RPiCameraDriver(width=640, height=480, prefer_picamera2=True)
        assert driver.backend_type == "picamera2"
        ret, frame = driver.read_frame()
        assert ret is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        driver.release()
        mock_inst.stop.assert_called_once()
        mock_inst.close.assert_called_once()

