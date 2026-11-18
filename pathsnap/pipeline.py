"""End-to-end prediction for one trip. OWNER: Sanjay.

This is the only function the API calls. It must never read the route file.
"""
import joblib

from pathsnap.config import ARTIFACTS_DIR


def load_model(path=ARTIFACTS_DIR / "model.joblib"):
    """Load the model bundle described in docs/CONTRACTS.md."""
    return joblib.load(path)


def predict_trip(trip, bundle):
    """Run clean -> candidates -> features -> classifier -> HMM on one trip.

    Returns a dict:
        pred_arc            int array, one arc id per GPS point (-1 = none)
        pred_arc_classifier int array, best classifier candidate per point (no HMM)
        timing_ms           dict of step name -> milliseconds
    """
    raise NotImplementedError("TODO(Sanjay)")
