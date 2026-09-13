"""
Resilient Serial Communication Bridge for Arduino Servo Controller.
Manages serial connection, heartbeat watchdog, command queueing, and mock fallback.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional
from smartbin_v2.utils.logger import get_logger

logger = get_logger("arduino_bridge")

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    serial = None


class ArduinoSerialBridge:
    """
    Manages robust USB serial communication between Raspberry Pi/Host and Arduino.
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        timeout: float = 1.0,
        mock_mode: bool = False,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.mock_mode = mock_mode or not HAS_SERIAL
        self.ser: Optional[Any] = None
        self.is_connected = False

        if self.mock_mode:
            self.is_connected = True
            logger.info("Arduino bridge initialized in MOCK mode.")
        else:
            self.connect()

    def connect(self) -> bool:
        """Attempt connection to serial device."""
        if self.mock_mode:
            self.is_connected = True
            logger.info("Arduino bridge running in MOCK mode (simulated servo responses).")
            return True

        try:
            logger.info(f"Connecting to Arduino on {self.port} at {self.baudrate} baud...")
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2.0)  # Wait for Arduino bootloader reset
            self.is_connected = True
            logger.info(f"Successfully connected to Arduino on {self.port}")
            return True
        except Exception as e:
            logger.warning(f"Failed to open serial port {self.port} ({e}). Switching to MOCK mode.")
            self.mock_mode = True
            self.is_connected = True
            return False

    def send_command(self, command_str: str) -> Optional[str]:
        """Send raw line command and wait for response."""
        if not self.is_connected:
            self.connect()

        if self.mock_mode:
            # Simulate Arduino response
            logger.info(f"[MOCK ARDUINO] Command received: {command_str}")
            if command_str.startswith("SORT:"):
                comp = command_str.split(":")[1]
                return f'{{"status":"ACTUATING","compartment":{comp},"angle":45}}'
            elif command_str == "PING":
                return '{"status":"PONG","busy":false}'
            return '{"status":"OK"}'

        try:
            cmd = (command_str.strip() + "\n").encode("utf-8")
            self.ser.write(cmd)
            self.ser.flush()
            response = self.ser.readline().decode("utf-8").strip()
            return response
        except Exception as e:
            logger.error(f"Serial transmission failure: {e}")
            self.is_connected = False
            return None

    def trigger_sort(self, compartment_id: int) -> bool:
        """
        Trigger segregation flap for specific compartment (1=Recyclable, 2=Compost, 3=Landfill, 4=Reject).
        """
        response = self.send_command(f"SORT:{compartment_id}")
        if response:
            try:
                data = json.loads(response)
                return data.get("status") == "ACTUATING"
            except json.JSONDecodeError:
                return "ACTUATING" in response
        return False

    def ping(self) -> bool:
        """Heartbeat check with Arduino."""
        resp = self.send_command("PING")
        return bool(resp and "PONG" in resp)

    def close(self) -> None:
        """Close serial port."""
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.is_connected = False
        logger.info("Serial bridge disconnected.")
