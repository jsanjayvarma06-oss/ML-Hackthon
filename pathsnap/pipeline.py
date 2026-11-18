"""End-to-end prediction for one trip. OWNER: Sanjay.

This is the only function the API calls. It must never read the route file.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pathsnap.candidates import K_CANDIDATES, RADIUS_M, clean_track, generate_candidates
from pathsnap.config import ARTIFACTS_DIR
from pathsnap.features import FEATURE_COLUMNS, extract_features
from pathsnap.geo import project_trip
from pathsnap.hmm import viterbi
from pathsnap.metrics import UNLABELLED


def load_model(path=ARTIFACTS_DIR / "model.joblib"):
    """Load the model bundle described in docs/CONTRACTS.md.

    Returns dict with keys: 'model', 'feature_columns', 'name', etc., or None if missing.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "model" not in bundle:
            raise ValueError(f"Invalid model format in {path}: expected dict with 'model' key.")
        return bundle
    except Exception as e:
        raise RuntimeError(f"Error loading model from {path}: {e}")


def predict_trip(trip, bundle=None):
    """Run clean -> candidates -> features -> classifier -> HMM on one trip.

    Returns a dict:
        pred_arc            int array, one arc id per GPS point (-1 = none)
        pred_arc_classifier int array, best classifier candidate per point (no HMM)
        timing_ms           dict of step name -> milliseconds
    """
    t0 = time.perf_counter()
    n_points = len(trip.track)

    # 1. Clean track
    t_clean_0 = time.perf_counter()
    keep_idx, _, _ = clean_track(trip.track)
    timing_clean = (time.perf_counter() - t_clean_0) * 1000.0

    # 2. Candidate generation
    t_cand_0 = time.perf_counter()
    txy = project_trip(trip)
    cand_arcs, cand_dist = generate_candidates(
        trip, k=K_CANDIDATES, radius_m=RADIUS_M, keep_idx=keep_idx, txy=txy
    )
    timing_cand = (time.perf_counter() - t_cand_0) * 1000.0

    # 3. Features and candidate probabilities (emission)
    t_feat_0 = time.perf_counter()
    pred_arc_classifier = np.full(n_points, UNLABELLED, dtype=int)
    emission_logp = [np.empty(0, dtype=float)] * n_points

    has_model = bundle is not None and isinstance(bundle, dict) and "model" in bundle

    if has_model:
        feat_records = extract_features(trip, cand_arcs, cand_dist, keep_idx=keep_idx, txy=txy)
        feat_cols = bundle.get("feature_columns", FEATURE_COLUMNS)

        if feat_records:
            feat_df = pd.DataFrame.from_records(feat_records)
            X = feat_df[feat_cols]
            model = bundle["model"]

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                # Binary classification: probability of label=1
                p1 = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                p1 = 1.0 / (1.0 + np.exp(-scores))
            else:
                p1 = np.asarray(model.predict(X), dtype=float)

            feat_df["prob"] = p1

            # Group probabilities and candidate ranks by point_idx
            for pt_idx, group in feat_df.groupby("point_idx"):
                p_vals = group["prob"].values.astype(float)
                arcs_subset = group["arc_id"].values.astype(int)

                # Classifier-only best candidate
                best_loc = int(np.argmax(p_vals))
                pred_arc_classifier[pt_idx] = arcs_subset[best_loc]

                # Normalized log probabilities for HMM emission
                p_sum = p_vals.sum()
                if p_sum > 0:
                    p_norm = p_vals / p_sum
                else:
                    p_norm = np.full(len(p_vals), 1.0 / len(p_vals))
                emission_logp[pt_idx] = np.log(np.maximum(p_norm, 1e-12))
    else:
        # Distance-only Gaussian emission fallback (Gaussian on dist_m, sigma=15m)
        for row in keep_idx:
            d = cand_dist[row]
            if len(d) > 0:
                p = np.exp(-0.5 * (d / 15.0) ** 2)
                p_sum = p.sum()
                p_norm = p / p_sum if p_sum > 0 else np.full(len(p), 1.0 / len(p))
                emission_logp[row] = np.log(np.maximum(p_norm, 1e-12))
                pred_arc_classifier[row] = cand_arcs[row][int(np.argmax(p_norm))]

    timing_feat = (time.perf_counter() - t_feat_0) * 1000.0

    # 4. HMM + Viterbi route decoding
    t_hmm_0 = time.perf_counter()
    pred_arc = viterbi(cand_arcs, emission_logp, trip)
    timing_hmm = (time.perf_counter() - t_hmm_0) * 1000.0

    total_time = (time.perf_counter() - t0) * 1000.0

    return {
        "pred_arc": pred_arc,
        "pred_arc_classifier": pred_arc_classifier,
        "timing_ms": {
            "clean": round(timing_clean, 2),
            "candidates": round(timing_cand, 2),
            "features_and_ml": round(timing_feat, 2),
            "hmm_viterbi": round(timing_hmm, 2),
            "total": round(total_time, 2),
        },
    }
