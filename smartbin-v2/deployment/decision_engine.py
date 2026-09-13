"""
Decision Engine and Actuation Broker for SmartBin AI v2.
Coordinates between AI pipeline decisions, physical Arduino servos, local audit logs,
offline storage, and external HTTP webhooks.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional
import requests

from smartbin_v2.arduino.serial_bridge import ArduinoSerialBridge
from smartbin_v2.deployment.offline_queue import OfflineEventQueue
from smartbin_v2.inference.pipeline import SegregationDecision
from smartbin_v2.utils.logger import DecisionLogger, get_logger

logger = get_logger("decision_engine")


class DecisionEngine:
    """
    Broker executing physical servo commands and audit telemetry for segregation decisions.
    """

    def __init__(
        self,
        arduino_bridge: Optional[ArduinoSerialBridge] = None,
        decision_log_path: str = "smartbin-v2/logs/decisions.jsonl",
        webhook_url: Optional[str] = None,
        offline_queue: Optional[OfflineEventQueue] = None,
    ) -> None:
        self.arduino = arduino_bridge or ArduinoSerialBridge(mock_mode=True)
        self.logger = DecisionLogger(decision_log_path)
        self.webhook_url = webhook_url
        self.offline_queue = offline_queue or OfflineEventQueue()
        self.last_actuation_time = 0.0
        self.cooldown_sec = 2.0  # Prevent back-to-back servo thrashing

    def dispatch_decision(self, decision: SegregationDecision) -> bool:
        """
        Execute physical actuation and record decision audit trail.
        """
        now = time.time()
        if now - self.last_actuation_time < self.cooldown_sec:
            logger.warning("Actuation suppressed by servo cooldown lock.")
            return False

        # 1. Trigger Physical Arduino Servo
        logger.info(
            f"Dispatching actuation to Compartment {decision.compartment_id} "
            f"({decision.target_bin.upper()} - {decision.class_name})"
        )
        servo_ok = self.arduino.trigger_sort(decision.compartment_id)
        self.last_actuation_time = now

        # 2. Build Event Payload
        event_record = {
            "target_bin": decision.target_bin,
            "compartment_id": decision.compartment_id,
            "class_name": decision.class_name,
            "class_id": decision.class_id,
            "confidence": decision.confidence,
            "material": decision.material,
            "is_contaminated": decision.is_contaminated,
            "track_id": decision.track_id,
            "actuated": servo_ok,
            "summary": decision.summary_message,
        }

        # 3. Log to local JSONL audit trail
        self.logger.log_decision(event_record)

        # 4. Enqueue to SQLite offline buffer
        self.offline_queue.enqueue(event_record)

        # 5. Dispatch Webhook if configured
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json=event_record, timeout=1.0)
            except Exception as e:
                logger.debug(f"Webhook dispatch notice: {e}")

        return servo_ok
