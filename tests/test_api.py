"""Backend test suite for PathSnap FastAPI server.

Tests:
1. GET /health endpoint.
2. GET /models endpoint.
3. GET /trips endpoint.
4. Invalid trip ID validation (path traversal, illegal characters).
5. Missing trip ID (HTTP 404).
6. Successful prediction (HTTP 200).
7. Prediction response schema adherence.
8. Ground-truth route security (reference route never exposed).
9. CORS headers for cross-laptop communication.
"""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ensure evaluation directory points to student_test for tests
os.environ["PATHSNAP_EVAL_DIR"] = str(ROOT / "data" / "raw" / "student_test")

from api.main import app
from pathsnap import io


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_1_health_endpoint(client):
    """Test 1: GET /health returns status ok and model information."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["api_version"] == "1.0.0"
    assert data["model_file_found"] is True


def test_2_models_endpoint(client):
    """Test 2: GET /models returns list of available model bundles."""
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) >= 1
    assert data["default"] is not None


def test_3_trips_endpoint(client):
    """Test 3: GET /trips returns authorised trip IDs without exposing routes."""
    resp = client.get("/trips")
    assert resp.status_code == 200
    data = resp.json()
    assert "trips" in data
    assert isinstance(data["trips"], list)
    assert len(data["trips"]) >= 10
    assert "00000000" in data["trips"]
    for item in data["trips"]:
        assert isinstance(item, str)


def test_4_invalid_trip_id(client):
    """Test 4: Invalid trip ID formats return HTTP 400."""
    invalid_ids = ["../00000000", "../../etc/passwd", "trip/123", ""]
    for bad_id in invalid_ids:
        resp = client.post("/predict", json={"trip_id": bad_id})
        assert resp.status_code in (400, 422)


def test_5_missing_trip(client):
    """Test 5: Non-existent trip ID returns HTTP 404."""
    resp = client.post("/predict", json={"trip_id": "99999999"})
    assert resp.status_code == 404


def test_6_successful_prediction(client):
    """Test 6: Successful prediction returns HTTP 200 and valid coordinates."""
    resp = client.post("/predict", json={"trip_id": "00000000"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["trip_id"] == "00000000"
    assert data["n_points"] > 0
    assert len(data["gps"]) == data["n_points"]
    assert len(data["predicted_path"]) > 0


def test_7_prediction_response_schema(client):
    """Test 7: Prediction response conforms to CONTRACTS.md and required schema."""
    resp = client.post("/predict", json={"trip_id": "00000000"})
    assert resp.status_code == 200
    data = resp.json()

    for key in ["success", "trip_id", "n_points", "gps", "predicted_path",
                "predicted_route", "metrics", "timing_ms", "processing_time_seconds", "message"]:
        assert key in data, f"Missing key: {key}"

    metrics = data["metrics"]
    assert metrics["route_continuity"] is not None
    assert metrics["route_continuity"] > 0.8


def test_8_ground_truth_never_returned(client):
    """Test 8: Reference route data is NEVER leaked in the API response."""
    resp = client.post("/predict", json={"trip_id": "00000000"})
    data = resp.json()

    assert "reference_route" not in data
    assert "true_route" not in data
    assert "ground_truth" not in data


def test_9_cors_headers(client):
    """Test 9: CORS headers are returned for cross-origin requests."""
    resp = client.options(
        "/predict",
        headers={
            "Origin": "http://192.168.1.50:8501",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"
