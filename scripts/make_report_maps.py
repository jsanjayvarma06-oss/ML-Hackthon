"""Report maps for a few train trips. OWNER: Sanjay.

Run from the repo root:  python scripts/make_report_maps.py
Writes results/maps/trip_<id>.html. Once artifacts/model.joblib exists and
pipeline.predict_trip works, the predicted route (red) is added automatically.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathsnap import io, pipeline, viz  # noqa: E402
from pathsnap.config import ARTIFACTS_DIR, RESULTS_DIR, TRAIN_DIR  # noqa: E402

# Train trips only: a short clean trip, one with many gaps, one with a dense network.
TRIPS = [(TRAIN_DIR, "00000005"), (TRAIN_DIR, "00000039"), (TRAIN_DIR, "00000046")]


def main():
    out_dir = RESULTS_DIR / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = None
    if (ARTIFACTS_DIR / "model.joblib").exists():
        bundle = pipeline.load_model()

    for part, trip_id in TRIPS:
        trip = io.load_trip(part, trip_id)
        route = io.load_route(part, trip_id)
        pred = None
        if bundle is not None:
            try:
                pred = pipeline.predict_trip(trip, bundle)["pred_arc"]
            except NotImplementedError:
                pass
        path = out_dir / f"trip_{trip_id}.html"
        viz.trip_map(trip, route=route, pred_arc=pred).save(str(path))
        print(f"{trip_id}: {len(trip.track)} points, predicted={'yes' if pred is not None else 'no'} -> {path}")


if __name__ == "__main__":
    main()
