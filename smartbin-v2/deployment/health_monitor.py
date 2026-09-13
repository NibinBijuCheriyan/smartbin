"""
Health Monitor and Watchdog Service for SmartBin AI Edge Devices.
Monitors CPU, Memory, SoC Thermal status, Frame Drops, and Camera Heartbeats.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional
import psutil

from smartbin_v2.raspberry_pi.benchmark_edge import get_soc_temperature
from smartbin_v2.utils.logger import get_logger

logger = get_logger("health_monitor")


@dataclass
class SystemHealthSnapshot:
    timestamp: float
    cpu_percent: float
    memory_used_mb: float
    memory_percent: float
    disk_free_gb: float
    temperature_celsius: float
    camera_alive: bool
    status: str


class HealthWatchdog:
    """
    Heartbeat and thermal watchdog maintaining system reliability.
    """

    def __init__(
        self,
        max_temp_celsius: float = 80.0,
        max_cpu_percent: float = 95.0,
        max_heartbeat_gap_sec: float = 5.0,
    ) -> None:
        self.max_temp = max_temp_celsius
        self.max_cpu = max_cpu_percent
        self.max_gap = max_heartbeat_gap_sec
        self.last_camera_heartbeat = time.time()

    def ping_camera_alive(self) -> None:
        """Register that a fresh frame was successfully processed."""
        self.last_camera_heartbeat = time.time()

    def check_health(self) -> SystemHealthSnapshot:
        """
        Inspect all vital hardware indicators.
        """
        now = time.time()
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/") if os.name != "nt" else psutil.disk_usage("C:\\")
        temp = get_soc_temperature()

        camera_alive = (now - self.last_camera_heartbeat) <= self.max_gap

        status = "HEALTHY"
        if temp > self.max_temp:
            status = "THERMAL_THROTTLING_WARNING"
            logger.warning(f"High Temperature Warning: {temp:.1f}°C")
        elif not camera_alive:
            status = "CAMERA_STREAM_FROZEN"
            logger.error("Camera stream appears frozen - no heartbeat received.")
        elif cpu > self.max_cpu:
            status = "HIGH_CPU_LOAD"

        return SystemHealthSnapshot(
            timestamp=now,
            cpu_percent=round(cpu, 1),
            memory_used_mb=round(mem.used / (1024 * 1024), 1),
            memory_percent=round(mem.percent, 1),
            disk_free_gb=round(disk.free / (1024 * 1024 * 1024), 2),
            temperature_celsius=round(temp, 1),
            camera_alive=camera_alive,
            status=status,
        )
