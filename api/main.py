"""FastAPI backend (Laptop 1). OWNER: Samarth.

Run from the repo root:
    export PATHSNAP_EVAL_DIR=data/raw/student_test   # demo day: the evaluator folder
    uvicorn api.main:app --host 0.0.0.0 --port 8000

The reference route is read only inside /predict to compute accuracy. It is never
passed to the model and never returned to the UI.
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pathsnap import io, labels, metrics, pipeline
from pathsnap.config import EVAL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap API")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = pipeline.load_model()
        log.info("model loaded: %s", _bundle.get("name"))
    return _bundle


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "eval_dir": str(EVAL_DIR)}


@app.get("/trips")
def trips():
    if not EVAL_DIR.exists():
        raise HTTPException(500, f"Trip folder not found: {EVAL_DIR}. Set PATHSNAP_EVAL_DIR.")
    ids = io.list_trips(EVAL_DIR)
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    log.info("POST /predict trip_id=%s", req.trip_id)
    if req.trip_id not in io.list_trips(EVAL_DIR):
        raise HTTPException(404, f"Unknown trip {req.trip_id}")
    t0 = time.perf_counter()
    trip = io.load_trip(EVAL_DIR, req.trip_id)
    try:
        out = pipeline.predict_trip(trip, get_bundle())
    except NotImplementedError as e:
        raise HTTPException(501, f"Pipeline not finished yet: {e}")

    # Scoring happens here, server side only.
    route = io.load_route(EVAL_DIR, req.trip_id)
    true_arc = labels.make_point_labels(trip, route)
    result = {
        "trip_id": req.trip_id,
        "n_points": int(len(trip.track)),
        "gps": [[float(lat), float(lon)] for lon, lat, _ in trip.track],
        "predicted_path": _arc_segments(trip, out["pred_arc"]),
        "metrics": {
            "top1_accuracy": metrics.top1_accuracy(true_arc, out["pred_arc"]),
            "top1_accuracy_classifier_only": metrics.top1_accuracy(true_arc, out["pred_arc_classifier"]),
            "route_continuity": metrics.route_continuity(out["pred_arc"], trip.arcs),
            "n_scored_points": int((true_arc != metrics.UNLABELLED).sum()),
        },
        "timing_ms": {**out.get("timing_ms", {}), "total": (time.perf_counter() - t0) * 1000},
    }
    log.info("trip %s top1=%.3f continuity=%.3f total_ms=%.0f", req.trip_id,
             result["metrics"]["top1_accuracy"], result["metrics"]["route_continuity"],
             result["timing_ms"]["total"])
    return result


def _arc_segments(trip, pred_arc):
    """Predicted arcs as [[lat, lon], [lat, lon]] line segments, consecutive repeats removed."""
    segs, prev = [], None
    for a in pred_arc:
        if a == metrics.UNLABELLED or a == prev:
            continue
        u, v = trip.arcs[a]
        (lon1, lat1), (lon2, lat2) = trip.nodes[u], trip.nodes[v]
        segs.append([[float(lat1), float(lon1)], [float(lat2), float(lon2)]])
        prev = a
    return segs
