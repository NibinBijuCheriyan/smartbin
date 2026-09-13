"""
Integration tests for FastAPI endpoints.
"""

import io
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from smartbin_v2.api.main import app

client = TestClient(app)


def test_api_health():
    """Verify /health endpoint returns 200 and valid keys."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "uptime_seconds" in data
    assert "cpu_percent" in data
    assert "ram_used_mb" in data


def test_api_metrics():
    """Verify /metrics endpoint returns telemetry statistics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_inferences" in data
    assert "class_breakdown" in data


def test_api_predict():
    """Verify /predict accepts image upload and returns valid decision."""
    # Generate test image in memory
    img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test.jpg", buf, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "target_bin" in data
    assert "compartment_id" in data
    assert "inference_latency_ms" in data
    assert data["target_bin"] in ["recyclable", "compost", "landfill", "reject"]
