"""
Persistent SQLite Offline Event Queue for SmartBin Edge Deployments.
Ensures zero data loss during network outages by buffering segregation records locally.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from smartbin_v2.utils.logger import get_logger

logger = get_logger("offline_queue")


class OfflineEventQueue:
    """
    Local SQLite database queue storing segregation records when remote APIs are unreachable.
    """

    def __init__(self, db_path: str | Path = "smartbin-v2/logs/offline_events.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    payload TEXT,
                    retry_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending'
                )
                """
            )
            conn.commit()

    def enqueue(self, event_data: Dict[str, Any]) -> int:
        """Push a decision event to local persistent queue."""
        payload_str = json.dumps(event_data)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO events (timestamp, payload, status) VALUES (?, ?, 'pending')",
                (time.time(), payload_str),
            )
            conn.commit()
            return cursor.lastrowid

    def get_pending(self, limit: int = 50) -> List[Tuple[int, Dict[str, Any]]]:
        """Fetch pending unsynced events."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, payload FROM events WHERE status = 'pending' ORDER BY id ASC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            return [(r[0], json.loads(r[1])) for r in rows]

    def mark_synced(self, event_id: int) -> None:
        """Mark event as successfully synced to cloud API."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE events SET status = 'synced' WHERE id = ?", (event_id,))
            conn.commit()

    def sync_to_cloud(self, cloud_endpoint: str, timeout: float = 3.0) -> int:
        """Sync all pending events to remote server."""
        pending = self.get_pending(limit=25)
        synced_count = 0

        for ev_id, payload in pending:
            try:
                resp = requests.post(cloud_endpoint, json=payload, timeout=timeout)
                if resp.status_code in [200, 201]:
                    self.mark_synced(ev_id)
                    synced_count += 1
            except Exception:
                # Still offline, break loop until next attempt
                break

        if synced_count > 0:
            logger.info(f"Synced {synced_count} offline events to cloud API.")
        return synced_count
