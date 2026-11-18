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
from pathsnap.config import ARTIFACTS_DIR, EVAL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap API")
_bundle = None
_extra = {}  # name -> bundle, for the optional models in artifacts/models/


MODEL_PATH = ARTIFACTS_DIR / "model.joblib"   # the final (default) model
MODELS_DIR = ARTIFACTS_DIR / "models"          # optional: one <name>.joblib per compared model


def get_bundle(name=None):
    """Return a model bundle, loading it once.

    name None (or the default model's name) -> artifacts/model.joblib, the chosen final model.
    Any other name -> artifacts/models/<name>.joblib, so the evaluator can compare models live.
    Raises 503 if the default file is missing and 404 for an unknown name.
    """
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                503, f"Model file not found: {MODEL_PATH}. Run scripts/export_model.py first.")
        _bundle = pipeline.load_model(MODEL_PATH)
        log.info("model loaded: %s", _bundle.get("name"))
    if name is None or name == _bundle.get("name"):
        return _bundle
    if name not in _extra:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise HTTPException(404, f"Unknown model {name}. See GET /models.")
        _extra[name] = pipeline.load_model(path)
        log.info("extra model loaded: %s", name)
    return _extra[name]


def available_models():
    """Default model name first, then every artifacts/models/<name>.joblib."""
    names = []
    if MODEL_PATH.exists():
        names.append(get_bundle().get("name"))
    if MODELS_DIR.exists():
        names += sorted(p.stem for p in MODELS_DIR.glob("*.joblib") if p.stem not in names)
    return names


@app.on_event("startup")
def _startup():
    log.info("eval_dir=%s (exists=%s) model_file=%s (exists=%s)",
             EVAL_DIR, EVAL_DIR.exists(), MODEL_PATH, MODEL_PATH.exists())
    # Load the model now, so the evaluator's first /predict is not slowed down by it.
    if MODEL_PATH.exists():
        get_bundle()


class PredictRequest(BaseModel):
    trip_id: str
    model: str | None = None  # optional; default = artifacts/model.joblib (the final model)


@app.get("/health")
def health():
    """Cheap liveness check. Reports the model name only if the model file exists."""
    name = None
    if _bundle is not None:
        name = _bundle.get("name")
    elif MODEL_PATH.exists():
        try:
            name = pipeline.load_model(MODEL_PATH).get("name")
        except Exception as e:  # corrupt file: say so instead of crashing /health
            log.warning("could not read model file: %s", e)
    return {"status": "ok", "eval_dir": str(EVAL_DIR), "model": name,
            "model_file_found": MODEL_PATH.exists()}


@app.get("/models")
def models():
    """Models the UI can choose from. The first one is the default (final) model."""
    names = available_models()
    return {"models": names, "default": names[0] if names else None}


@app.get("/trips")
def trips():
    """List the trip ids available in PATHSNAP_EVAL_DIR."""
    if not EVAL_DIR.exists():
        raise HTTPException(500, f"Trip folder not found: {EVAL_DIR}. Set PATHSNAP_EVAL_DIR.")
    ids = io.list_trips(EVAL_DIR)
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    """Run the full pipeline on one trip and return GPS, predicted path, metrics and timing."""
    log.info("POST /predict trip_id=%s model=%s", req.trip_id, req.model or "default")
    if not EVAL_DIR.exists():
        raise HTTPException(500, f"Trip folder not found: {EVAL_DIR}. Set PATHSNAP_EVAL_DIR.")
    bundle = get_bundle(req.model)
    if req.trip_id not in io.list_trips(EVAL_DIR):
        raise HTTPException(404, f"Unknown trip {req.trip_id}")
    t0 = time.perf_counter()
    trip = io.load_trip(EVAL_DIR, req.trip_id)
    try:
        out = pipeline.predict_trip(trip, bundle)
    except NotImplementedError as e:
        raise HTTPException(501, f"Pipeline not finished yet: {e}")

    # Scoring happens here, server side only.
    route = io.load_route(EVAL_DIR, req.trip_id)
    true_arc = labels.make_point_labels(trip, route)
    result = {
        "trip_id": req.trip_id,
        "model": bundle.get("name"),
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
    log.info("trip %s model=%s top1=%.3f continuity=%.3f total_ms=%.0f", req.trip_id, result["model"],
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
