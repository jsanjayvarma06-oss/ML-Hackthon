"""Train and compare the 5+ models. OWNER: Kushal.

Run from the repo root:  python scripts/train_models.py
- GroupKFold on train.parquet, grouped by trip_id (never a random split)
- Fit on all train, evaluate on student_test.parquet
- Writes results/models/model_comparison.csv (CV macro F1, student-test top-1, macro F1,
  log loss, train time, inference time) and comparison plots in results/models/plots/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathsnap import features, metrics, models  # noqa: E402
from pathsnap.config import PROCESSED_DIR, RESULTS_DIR, SEED  # noqa: E402


def main():
    raise NotImplementedError("TODO(Kushal)")


if __name__ == "__main__":
    main()
