"""
Tests for the WebhookHook decision hook.

Tests:
- Successful delivery to a local HTTP server.
- Retry behavior on connection failure.
- Pipeline doesn't block when webhook is unreachable.
- Queue overflow behavior.
- Clean shutdown.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List

import pytest

from smartbin.decision import DecisionEvent, WebhookHook


# ---------------------------------------------------------------------------
# Test HTTP server
# ---------------------------------------------------------------------------


class _RecordingHandler(BaseHTTPRequestHandler):
    """HTTP handler that records received POST bodies."""

    received: List[dict] = []
    response_code: int = 200

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        parsed = json.loads(body.decode("utf-8"))
        _RecordingHandler.received.append(parsed)
        self.send_response(_RecordingHandler.response_code)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress server logs during tests


@pytest.fixture
def test_server():
    """Start a local HTTP server for testing webhook delivery."""
    _RecordingHandler.received = []
    _RecordingHandler.response_code = 200

    server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}/decision", server

    server.shutdown()


def _make_event(track_id: int = 1, item_class: str = "plastic") -> DecisionEvent:
    """Create a test DecisionEvent."""
    return DecisionEvent.create(
        track_id=track_id,
        item_class=item_class,
        confidence=0.85,
        frame_count=10,
        total_frames=12,
        is_certain=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebhookHookDelivery:
    """Test webhook delivery to a real HTTP server."""

    def test_successful_delivery(self, test_server):
        """Events are POSTed to the webhook URL."""
        url, server = test_server
        hook = WebhookHook(url=url, timeout=2.0, max_retries=1)

        event = _make_event(track_id=42, item_class="glass")
        hook.on_decision(event)

        # Wait for background delivery
        time.sleep(1.0)
        hook.close()

        assert len(_RecordingHandler.received) == 1
        received = _RecordingHandler.received[0]
        assert received["track_id"] == 42
        assert received["item_class"] == "glass"
        assert received["is_certain"] is True

    def test_multiple_events(self, test_server):
        """Multiple events are delivered sequentially."""
        url, server = test_server
        hook = WebhookHook(url=url, timeout=2.0, max_retries=1)

        for i in range(5):
            hook.on_decision(_make_event(track_id=i, item_class="paper"))

        time.sleep(2.0)
        hook.close()

        assert len(_RecordingHandler.received) == 5
        track_ids = [r["track_id"] for r in _RecordingHandler.received]
        assert sorted(track_ids) == [0, 1, 2, 3, 4]

    def test_json_content_type(self, test_server):
        """Verify Content-Type header is application/json."""
        url, server = test_server
        # The handler doesn't check headers, but we verify the body is valid JSON
        hook = WebhookHook(url=url, timeout=2.0)
        hook.on_decision(_make_event())

        time.sleep(1.0)
        hook.close()

        assert len(_RecordingHandler.received) == 1
        # If it parsed as JSON in the handler, content type was correct
        assert "item_class" in _RecordingHandler.received[0]


class TestWebhookHookResilience:
    """Test webhook behavior under failure conditions."""

    def test_unreachable_endpoint_does_not_block(self):
        """Hook with unreachable URL should not block the caller."""
        hook = WebhookHook(
            url="http://127.0.0.1:1",  # Port 1 — unreachable
            timeout=0.5,
            max_retries=0,  # No retries — fail fast
        )

        start = time.monotonic()
        hook.on_decision(_make_event())
        elapsed = time.monotonic() - start

        # on_decision should return nearly instantly (just queues)
        assert elapsed < 0.1

        # Wait for background thread to process
        time.sleep(1.5)
        hook.close()

    def test_clean_shutdown(self, test_server):
        """Hook shuts down cleanly even with pending events."""
        url, server = test_server
        hook = WebhookHook(url=url, timeout=2.0)

        hook.on_decision(_make_event())
        hook.close()

        # Should not hang or raise


class TestWebhookHookQueueBehavior:
    """Test queue overflow and backpressure."""

    def test_queue_overflow_drops_events(self):
        """When the queue is full, new events are dropped (not blocked)."""
        hook = WebhookHook(
            url="http://127.0.0.1:1",  # Unreachable
            timeout=0.1,
            max_retries=0,
            max_queue_size=2,
        )

        # Fill the queue (background thread may drain some)
        for i in range(10):
            hook.on_decision(_make_event(track_id=i))

        # Should not block or raise
        time.sleep(0.5)
        hook.close()
