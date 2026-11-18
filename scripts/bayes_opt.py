"""Bayesian hyperparameter optimisation of the chosen model (Optuna). OWNER: Kushal.

Run from the repo root:  python scripts/bayes_opt.py
- Objective: mean GroupKFold score on train only (never the evaluator data)
- Writes results/models/optuna_trials.csv, results/models/best_params.json,
  results/models/tuned_vs_untuned.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathsnap.config import PROCESSED_DIR, RESULTS_DIR, SEED  # noqa: E402


def main():
    raise NotImplementedError("TODO(Kushal)")


if __name__ == "__main__":
    main()
