"""End-to-end prediction for one trip. OWNER: Sanjay.

This is the only function the API calls. It must never read the route file.
"""
import time

import joblib
import numpy as np

from pathsnap import features, hmm
from pathsnap.config import ARTIFACTS_DIR


def load_model(path=ARTIFACTS_DIR / "model.joblib"):
    """Load the model bundle described in docs/CONTRACTS.md."""
    return joblib.load(path)


def split_by_point(df, values, cand_arcs, keep_idx):
    """Cut a per-candidate-row array into one array per original track row.

    df: candidate table from features.trip_table (rows ordered by kept point, then rank).
    values: array with one entry per row of df. cand_arcs / keep_idx as from trip_table.
    Returns a list with one array per original track row (empty where no candidates).
    Used to line classifier probabilities up with the candidate lists the HMM works on.
    """
    out = [np.empty(0)] * len(cand_arcs)
    pos = 0
    for r in keep_idx:
        n = len(cand_arcs[r])
        if n:
            out[r] = np.asarray(values[pos:pos + n], dtype=float)
            pos += n
    return out


def predict_trip(trip, bundle):
    """Run clean -> candidates -> features -> classifier -> HMM on one trip.

    trip: io.Trip (track, nodes, arcs; never the route).
    bundle: model bundle dict (docs/CONTRACTS.md) or None. With None (or a dict without a
    "model") the emission is the distance-only Gaussian, so the whole pipeline still runs without a trained model.
    bundle["hmm_params"], if present, overrides the HMM defaults.

    Returns a dict:
        pred_arc            int array, one arc id per GPS point (-1 = none)
        pred_arc_classifier int array, best classifier candidate per point (no HMM)
        timing_ms           dict of step name -> milliseconds
    """
    t0 = time.perf_counter()
    df, cand_arcs, cand_dist, keep_idx, segment_id, txy = features.trip_table(trip)
    t1 = time.perf_counter()
    params = (bundle or {}).get("hmm_params")

    if bundle is not None and "model" in bundle and len(df):
        cols = bundle["feature_columns"]
        prob = bundle["model"].predict_proba(df[cols])[:, 1]
        emission = hmm.classifier_emission(split_by_point(df, prob, cand_arcs, keep_idx), params)
    else:
        emission = hmm.distance_emission(cand_dist, params)
    t2 = time.perf_counter()

    n = len(trip.track)
    pred_clf = hmm.argmax_arcs(cand_arcs, emission, keep_idx, n)
    pred = hmm.viterbi(cand_arcs, emission, trip, txy, keep_idx, segment_id, params)
    t3 = time.perf_counter()
    return {
        "pred_arc": pred,
        "pred_arc_classifier": pred_clf,
        "timing_ms": {
            "candidates_features": (t1 - t0) * 1000,
            "classifier": (t2 - t1) * 1000,
            "hmm": (t3 - t2) * 1000,
            "total": (t3 - t0) * 1000,
        },
    }
