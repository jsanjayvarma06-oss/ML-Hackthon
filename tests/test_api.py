"""Backend test suite for PathSnap FastAPI server.

Tests:
1. Server startup and lifespan initialization.
2. GET /health endpoint.
3. GET /trips endpoint.
4. Invalid trip ID validation (path traversal, illegal characters).
5. Missing trip ID (HTTP 404).
6. Model unavailable fallback handling.
7. Successful prediction (HTTP 200).
8. Prediction response schema adherence.
9. Ground-truth route security (reference route never exposed).
10. Multiple sequential requests (state independence).
11. Malformed request error handling.
12. CORS headers for cross-laptop communication.
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

from api.main import app, _state
from pathsnap import io


@pytest.fixture(scope="module")
def client():
    """Create test client with lifespan context."""
    with TestClient(app) as c:
        yield c


def test_1_server_startup(client):
    """Test 1: Server startup and lifespan initialization."""
    assert _state["version"] == "1.0.0"
    assert _state["model_loaded"] is True
    assert _state["bundle"] is not None


def test_2_health_endpoint(client):
    """Test 2: GET /health returns server status, model status, and dataset availability."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["api_version"] == "1.0.0"
    assert data["model_loaded"] is True
    assert data["model_name"] is not None
    assert data["data_available"] is True
    assert data["evaluation_trips_count"] >= 10


def test_3_trips_endpoint(client):
    """Test 3: GET /trips returns authorised trip IDs without exposing routes."""
    resp = client.get("/trips")
    assert resp.status_code == 200
    data = resp.json()
    assert "trips" in data
    assert isinstance(data["trips"], list)
    assert len(data["trips"]) >= 10
    assert "00000000" in data["trips"]
    # Verify no route contents are exposed in /trips
    for item in data["trips"]:
        assert isinstance(item, str)
        assert not isinstance(item, list)


def test_4_invalid_trip_id(client):
    """Test 4: Invalid trip ID formats (directory traversal, special chars) return HTTP 400."""
    invalid_ids = ["../00000000", "../../etc/passwd", "trip/123", "trip;drop table", "a" * 100, ""]
    for bad_id in invalid_ids:
        resp = client.post("/predict", json={"trip_id": bad_id})
        assert resp.status_code in (400, 422)


def test_5_missing_trip(client):
    """Test 5: Non-existent trip ID returns HTTP 404."""
    resp = client.post("/predict", json={"trip_id": "99999999"})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_6_model_unavailable_fallback(client):
    """Test 6: Model unavailable handles prediction gracefully via distance emission."""
    saved_bundle = _state["bundle"]
    try:
        _state["bundle"] = None
        resp = client.post("/predict", json={"trip_id": "00000000"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["metrics"]["route_continuity"] is not None
        assert data["metrics"]["route_continuity"] > 0.8
    finally:
        _state["bundle"] = saved_bundle


def test_7_successful_prediction(client):
    """Test 7: Successful prediction on valid trip returns HTTP 200 and valid path."""
    resp = client.post("/predict", json={"trip_id": "00000000"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["trip_id"] == "00000000"
    assert data["n_points"] > 0
    assert len(data["gps"]) == data["n_points"]
    assert len(data["predicted_path"]) > 0


def test_8_prediction_response_schema(client):
    """Test 8: Prediction response schema conforms to CONTRACTS.md and prompt requirements."""
    resp = client.post("/predict", json={"trip_id": "00000000"})
    assert resp.status_code == 200
    data = resp.json()

    # Required top-level keys
    for key in ["success", "trip_id", "n_points", "gps", "predicted_path",
                "predicted_route", "metrics", "timing_ms", "processing_time_seconds", "message"]:
        assert key in data, f"Missing key: {key}"

    # Verify metrics sub-dictionary
    metrics = data["metrics"]
    for m in ["top1_accuracy", "top1_accuracy_classifier_only", "route_continuity", "macro_f1", "n_scored_points"]:
        assert m in metrics, f"Missing metric: {m}"
    assert metrics["top1_accuracy"] > 0.8
    assert metrics["route_continuity"] > 0.9

    # Verify timing sub-dictionary
    timing = data["timing_ms"]
    for t in ["clean", "candidates", "features_and_ml", "hmm_viterbi", "total"]:
        assert t in timing, f"Missing timing: {t}"

    # Verify coordinate format: [[lat, lon], ...]
    assert len(data["gps"][0]) == 2
    # Verify line segment format: [[[lat, lon], [lat, lon]], ...]
    assert len(data["predicted_path"][0]) == 2
    assert len(data["predicted_path"][0][0]) == 2


def test_9_ground_truth_never_returned(client):
    """Test 9: Reference route data is NEVER leaked to the client."""
    eval_dir = Path(os.environ["PATHSNAP_EVAL_DIR"])
    route = io.load_route(eval_dir, "00000000")
    route_list = route.tolist()

    resp = client.post("/predict", json={"trip_id": "00000000"})
    data = resp.json()

    # The exact ground-truth sequence must not be sent as a field
    assert "reference_route" not in data
    assert "true_route" not in data
    assert "ground_truth" not in data
    assert "labels" not in data


def test_10_multiple_sequential_requests(client):
    """Test 10: Multiple sequential requests execute independently without memory leakage."""
    trips_to_test = ["00000000", "00000007"]
    for tid in trips_to_test:
        resp = client.post("/predict", json={"trip_id": tid})
        assert resp.status_code == 200
        data = resp.json()
        assert data["trip_id"] == tid
        assert data["success"] is True


def test_11_error_handling_malformed_body(client):
    """Test 11: Malformed JSON or missing fields return HTTP 422 unprocessable entity."""
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_12_cors_headers(client):
    """Test 12: CORS headers are properly returned for cross-origin / cross-laptop requests."""
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
