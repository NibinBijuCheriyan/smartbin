"""
Industrial structured logging module for SmartBin AI v2.
Supports standard console logging, rotating file logs, and JSONL decision telemetry.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON strings for ingestion into ELK/Grafana/Datadog."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if hasattr(record, "extra_data"):
            log_obj["data"] = record.extra_data
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Configure the root logger with console and optional file handlers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to save text logs.
        json_format: If True, output structured JSON.
    """
    root_logger = logging.getLogger("smartbin")
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if re-initialized
    if root_logger.handlers:
        return root_logger

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        fmt = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s"
        console_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root_logger.addHandler(console_handler)

    # File Handler
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            JSONFormatter() if json_format else logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s"
            )
        )
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger inheriting smartbin root configuration."""
    return logging.getLogger(f"smartbin.{name}")


class DecisionLogger:
    """Thread-safe JSONL decision logger for recording waste sorting outcomes."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_decision(self, decision_data: Dict[str, Any]) -> None:
        """Append an actuation/segregation decision event as JSON line."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **decision_data,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
