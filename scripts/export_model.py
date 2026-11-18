"""Refit the tuned model on all train trips and export the model bundle. OWNER: Kushal.

Run from the repo root (after scripts/bayes_opt.py):
    python scripts/export_model.py

Reads results/models/best_params.json, rebuilds the estimator with models.build, fits it
on all 80 train trips (labelled rows), and writes artifacts/model.joblib, the dict that
Sanjay's pipeline.predict_trip and the API load (docs/CONTRACTS.md section 2):
    {model, feature_columns, name, params, trained_on, hmm_params}
hmm_params comes from results/pipeline/hmm_params.json if it exists, else None.
Then reloads the file and checks it gives identical probabilities (guards against a
broken or stale artifact). student_test is not touched.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_models as tm  # noqa: E402
from pathsnap import features, models  # noqa: E402
from pathsnap.config import ARTIFACTS_DIR, PROCESSED_DIR, RESULTS_DIR  # noqa: E402


def main():
    """Fit on all train rows, dump the bundle, verify the reload."""
    best = json.loads((tm.OUT / "best_params.json").read_text())
    name, params = best["model"], best["params"]
    cols = list(features.FEATURE_COLUMNS)
    train = tm.labelled(pd.read_parquet(PROCESSED_DIR / "train.parquet"))
    sub = models.TRAIN_SUBSAMPLE.get(name)
    if sub:
        train = train.loc[tm.subsample_points(train, np.arange(len(train)), sub)]
    est = tm.fit(models.build(name, params), train[cols], train["label"])

    hp_path = RESULTS_DIR / "pipeline" / "hmm_params.json"
    hmm_params = json.loads(hp_path.read_text()) if hp_path.exists() else None
    bundle = {
        "model": est, "feature_columns": cols, "name": f"{name}_tuned", "params": params,
        "trained_on": f"train ({train.trip_id.nunique()} trips, {len(train):,} labelled rows)",
        "hmm_params": hmm_params,
    }
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(bundle, out, compress=3)

    back = joblib.load(out)
    sample = train[cols].iloc[:20000]
    same = np.array_equal(bundle["model"].predict_proba(sample)[:, 1],
                          back["model"].predict_proba(sample)[:, 1])
    assert same and back["feature_columns"] == cols, "reloaded model differs"
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB); reload gives identical "
          f"probabilities on {len(sample):,} rows; hmm_params={'set' if hmm_params else None}")


if __name__ == "__main__":
    main()
