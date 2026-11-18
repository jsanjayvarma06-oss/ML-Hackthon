"""FastAPI backend (Laptop 1). OWNER: Samarth / Jagdeesh.

PathSnap - Distributed Map-Matching & Route Reconstruction Backend.

Run from the repo root:
    export PATHSNAP_EVAL_DIR=data/raw/student_test   # or data/evaluator on demo day
    uvicorn api.main:app --host 0.0.0.0 --port 8000

The reference route is read only inside /predict to compute metrics on the server.
It is never passed to the model, never logged, and never returned to the client.
"""
import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import f1_score

from pathsnap import io, labels, metrics, pipeline
from pathsnap.config import ARTIFACTS_DIR, EVAL_DIR

# Configure clean, distinct console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pathsnap.server")

# Global model bundle and server state
_state: Dict[str, Any] = {
    "bundle": None,
    "model_loaded": False,
    "model_name": None,
    "model_error": None,
    "version": "1.0.0",
}

TRIP_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once during startup and verify evaluation dataset."""
    print("[SERVER] PathSnap backend starting...", flush=True)

    # 1. Load trained model artifact
    model_path = ARTIFACTS_DIR / "model.joblib"
    try:
        if model_path.exists():
            bundle = pipeline.load_model(model_path)
            _state["bundle"] = bundle
            _state["model_loaded"] = True
            _state["model_name"] = bundle.get("name", "unnamed_model")
            print(
                f"[MODEL] Model loaded successfully: {_state['model_name']} "
                f"(trained on: {bundle.get('trained_on', 'unknown')})",
                flush=True,
            )
        else:
            _state["model_loaded"] = False
            _state["model_error"] = f"Model artifact not found at {model_path.name}"
            print(
                f"[MODEL] [WARNING] {model_path} not found. Fallback to distance-based emission available.",
                flush=True,
            )
    except Exception as e:
        _state["model_loaded"] = False
        _state["model_error"] = str(e)
        print(f"[MODEL] [ERROR] Failed to load model bundle: {e}", flush=True)

    # 2. Check evaluation data directory
    if EVAL_DIR.exists():
        trips = io.list_trips(EVAL_DIR)
        print(f"[DATA] Evaluation dataset verified: {len(trips)} trips available in {EVAL_DIR.name}", flush=True)
    else:
        print(f"[DATA] [WARNING] Evaluation directory does not exist: {EVAL_DIR.name}", flush=True)

    print("[NETWORK] Server listening on 0.0.0.0:8000 (accessible across LAN / Wi-Fi)", flush=True)
    yield
    print("[SERVER] PathSnap backend shutting down...", flush=True)


app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed map-matching and route reconstruction server",
    version=_state["version"],
    lifespan=lifespan,
)

# Enable CORS for two-laptop communication (Laptop 2 Streamlit UI / Web Client)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="ID of the trip to reconstruct, e.g. '00000000'")


def _validate_trip_id(trip_id: str) -> str:
    """Validate trip ID to prevent directory traversal and invalid input."""
    if not trip_id or not TRIP_ID_REGEX.match(trip_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid trip_id format: '{trip_id}'. Only alphanumeric characters, dashes, and underscores allowed.",
        )
    return trip_id


@app.get("/health")
def health(request: Request):
    """Health endpoint returning server status, model loading status, data availability and API version."""
    client_ip = request.client.host if request.client else "unknown"
    print(f"[CLIENT] Incoming request: GET /health from {client_ip}", flush=True)

    data_available = EVAL_DIR.exists()
    n_trips = len(io.list_trips(EVAL_DIR)) if data_available else 0

    return {
        "status": "healthy",
        "api_version": _state["version"],
        "model_loaded": _state["model_loaded"],
        "model_name": _state["model_name"],
        "model_error": _state["model_error"],
        "data_available": data_available,
        "evaluation_trips_count": n_trips,
        "eval_dir": EVAL_DIR.name,
    }


@app.get("/trips")
def trips(request: Request):
    """List authorised evaluation trip IDs available on the server. Never exposes reference routes."""
    client_ip = request.client.host if request.client else "unknown"
    print(f"[CLIENT] Incoming request: GET /trips from {client_ip}", flush=True)

    if not EVAL_DIR.exists():
        print(f"[DATA] [ERROR] Evaluation folder not found: {EVAL_DIR.name}", flush=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trip folder not found: {EVAL_DIR.name}. Set PATHSNAP_EVAL_DIR environment variable.",
        )

    available_trips = io.list_trips(EVAL_DIR)
    print(f"[RESPONSE] GET /trips -> {len(available_trips)} authorised trips returned", flush=True)
    return {"trips": available_trips}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    """Process selected GPS trip, perform ML prediction + HMM Viterbi, score server-side, and return results."""
    client_ip = request.client.host if request.client else "unknown"
    t_start = time.perf_counter()
    trip_id = _validate_trip_id(req.trip_id)

    print(f"[CLIENT] Incoming request: POST /predict from {client_ip} for trip {trip_id}", flush=True)

    if not EVAL_DIR.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trip directory '{EVAL_DIR.name}' not found on server.",
        )

    available_trips = io.list_trips(EVAL_DIR)
    if trip_id not in available_trips:
        print(f"[TRIP] [ERROR] Trip '{trip_id}' not found in available evaluation trips", flush=True)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip '{trip_id}' not found in evaluation dataset.",
        )

    # 1. Load GPS track from server-side dataset
    trip = io.load_trip(EVAL_DIR, trip_id)
    n_points = len(trip.track)
    print(f"[TRIP] Processing trip: {trip_id} ({n_points} raw GPS observations, {len(trip.arcs)} road arcs)", flush=True)

    # 2. Run ML + HMM route reconstruction pipeline
    try:
        pipeline_out = pipeline.predict_trip(trip, _state.get("bundle"))
    except Exception as e:
        print(f"[PIPELINE] [ERROR] Pipeline failure on trip {trip_id}: {e}", flush=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failed: {e}",
        )

    timing = pipeline_out.get("timing_ms", {})
    print(f"[ML] Candidate generation completed in {timing.get('candidates', 0.0):.1f} ms", flush=True)
    print(f"[ML] Classifier prediction completed in {timing.get('features_and_ml', 0.0):.1f} ms", flush=True)
    print(f"[HMM] Viterbi decoding completed in {timing.get('hmm_viterbi', 0.0):.1f} ms", flush=True)

    pred_arc = pipeline_out["pred_arc"]
    pred_arc_classifier = pipeline_out["pred_arc_classifier"]

    # 3. Server-side evaluation against reference route (if route file exists)
    route_file = EVAL_DIR / trip_id / f"{trip_id}.route"
    metrics_dict: Dict[str, Optional[float]] = {
        "top1_accuracy": None,
        "top1_accuracy_classifier_only": None,
        "macro_f1": None,
        "log_loss": None,
        "route_continuity": metrics.route_continuity(pred_arc, trip.arcs),
        "n_scored_points": 0,
    }

    if route_file.exists():
        try:
            route = io.load_route(EVAL_DIR, trip_id)
            true_arc = labels.make_point_labels(trip, route)
            valid_mask = true_arc != metrics.UNLABELLED
            n_scored = int(valid_mask.sum())
            metrics_dict["n_scored_points"] = n_scored

            if n_scored > 0:
                top1 = metrics.top1_accuracy(true_arc, pred_arc)
                top1_clf = metrics.top1_accuracy(true_arc, pred_arc_classifier)
                metrics_dict["top1_accuracy"] = round(top1, 4) if top1 is not None else None
                metrics_dict["top1_accuracy_classifier_only"] = round(top1_clf, 4) if top1_clf is not None else None

                # Server-side macro F1 across matched arcs
                y_true_valid = true_arc[valid_mask]
                y_pred_valid = pred_arc[valid_mask]
                f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
                metrics_dict["macro_f1"] = round(float(f1), 4)

            print(
                f"[EVALUATION] Metrics calculated server-side: top1={metrics_dict['top1_accuracy']}, "
                f"continuity={metrics_dict['route_continuity']:.4f}, scored_points={n_scored}",
                flush=True,
            )
        except Exception as eval_err:
            print(f"[EVALUATION] [WARNING] Server evaluation error: {eval_err}", flush=True)
    else:
        print("[EVALUATION] Reference route not available on server; skipping ground-truth evaluation", flush=True)

    # 4. Reconstruct predicted road route geometries and arc list
    predicted_path = _arc_segments(trip, pred_arc)
    predicted_route_arcs = [int(a) for a in pred_arc if a != metrics.UNLABELLED]

    total_time_s = time.perf_counter() - t_start
    timing["total"] = round(total_time_s * 1000.0, 2)

    response_data = {
        "success": True,
        "trip_id": trip_id,
        "n_points": n_points,
        "gps": [[float(lat), float(lon)] for lon, lat, _ in trip.track],
        "predicted_path": predicted_path,
        "predicted_route": predicted_route_arcs,
        "metrics": metrics_dict,
        "timing_ms": timing,
        "processing_time_seconds": round(total_time_s, 4),
        "message": "Prediction completed successfully",
    }

    print(
        f"[RESPONSE] Result returned successfully in {total_time_s:.3f}s (HTTP 200)",
        flush=True,
    )
    return response_data


def _arc_segments(trip, pred_arc):
    """Convert predicted arcs to [[lat, lon], [lat, lon]] line segments with consecutive repeats removed."""
    segs, prev = [], None
    for a in pred_arc:
        if a == metrics.UNLABELLED or a == prev:
            continue
        u, v = trip.arcs[a]
        (lon1, lat1), (lon2, lat2) = trip.nodes[u], trip.nodes[v]
        segs.append([[float(lat1), float(lon1)], [float(lat2), float(lon2)]])
        prev = a
    return segs
