"""Refit the tuned model on all train trips and export the model bundle. OWNER: Kushal.

Run from the repo root (after scripts/bayes_opt.py):
    python scripts/export_model.py

Reads results/models/best_params.json, rebuilds the estimator with models.build, fits it
on all 80 train trips (labelled rows), and writes artifacts/model.joblib, the dict that
Sanjay's pipeline.predict_trip and the API load (docs/CONTRACTS.md section 2):
    {model, feature_columns, name, params, trained_on, hmm_params}
hmm_params comes from results/pipeline/hmm_params.json if it exists, else None.
Then reloads the file and checks it gives identical probabilities (guards against a
broken or stale artifact). It also writes artifacts/models/<name>.joblib, one untuned
bundle per compared model, for the UI model picker; any bundle over 50 MB is skipped
and reported. student_test is not touched.
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


MAX_BUNDLE_MB = 50  # GitHub rejects files > 100 MB; bigger bundles are skipped and reported


def load_hmm_params():
    """HMM settings for the bundle: results/pipeline/hmm_params.json 'classifier_hmm' entry
    if present (else the whole file), or None when Sanjay's file does not exist yet."""
    hp_path = RESULTS_DIR / "pipeline" / "hmm_params.json"
    if not hp_path.exists():
        return None
    hp = json.loads(hp_path.read_text())
    return hp.get("classifier_hmm", hp)


def fit_bundle(name, params, label, train, cols, hmm_params):
    """Fit model `name` (params dict) on all given train rows and return the bundle dict.

    Slow models (models.TRAIN_SUBSAMPLE) are fitted on the same fixed point subsample as
    in cross-validation. Contract: docs/CONTRACTS.md section 2.
    """
    sub = models.TRAIN_SUBSAMPLE.get(name)
    if sub:
        train = train.loc[tm.subsample_points(train, np.arange(len(train)), sub)]
    est = tm.fit(models.build(name, params), train[cols], train["label"])
    return {
        "model": est, "feature_columns": cols, "name": label, "params": params,
        "trained_on": f"train ({train.trip_id.nunique()} trips, {len(train):,} labelled rows"
                      + (f", subsample of {sub:,} points" if sub else "") + ")",
        "hmm_params": hmm_params,
    }, train


def save_and_verify(bundle, train, cols, out):
    """Dump bundle to `out` and check the reload gives identical probabilities.

    Returns size in MB."""
    joblib.dump(bundle, out, compress=3)
    back = joblib.load(out)
    sample = train[cols].iloc[:20000]
    same = np.array_equal(bundle["model"].predict_proba(sample)[:, 1],
                          back["model"].predict_proba(sample)[:, 1])
    assert same and back["feature_columns"] == cols, f"reloaded {out.name} differs"
    return out.stat().st_size / 1e6


def main():
    """Export the tuned default bundle, then one untuned bundle per compared model."""
    best = json.loads((tm.OUT / "best_params.json").read_text())
    cols = list(features.FEATURE_COLUMNS)
    train = tm.labelled(pd.read_parquet(PROCESSED_DIR / "train.parquet"))
    hmm_params = load_hmm_params()
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    bundle, used = fit_bundle(best["model"], best["params"], f"{best['model']}_tuned",
                              train, cols, hmm_params)
    out = ARTIFACTS_DIR / "model.joblib"
    mb = save_and_verify(bundle, used, cols, out)
    print(f"Wrote {out} ({mb:.1f} MB), reload identical, hmm_params="
          f"{'set' if hmm_params else None}", flush=True)

    # per-model bundles for the UI model picker (untuned defaults, same fit as the CV table)
    (ARTIFACTS_DIR / "models").mkdir(exist_ok=True)
    skipped = []
    for name in models.get_models():
        out = ARTIFACTS_DIR / "models" / f"{name}.joblib"
        b, used = fit_bundle(name, {}, name, train, cols, hmm_params)
        mb = save_and_verify(b, used, cols, out)
        if mb > MAX_BUNDLE_MB:
            out.unlink()
            skipped.append((name, mb))
            print(f"SKIPPED {name}: {mb:.0f} MB > {MAX_BUNDLE_MB} MB", flush=True)
        else:
            print(f"Wrote {out} ({mb:.1f} MB), reload identical", flush=True)
    if skipped:
        print("Skipped (too large):", skipped)


if __name__ == "__main__":
    main()
