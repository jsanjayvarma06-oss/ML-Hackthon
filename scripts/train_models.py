"""Train and compare the 5+ models. OWNER: Kushal.

Run from the repo root:  python scripts/train_models.py
- GroupKFold on train.parquet, grouped by trip_id (never a random split)
- Writes results/models/model_comparison.csv (CV macro F1, CV top-1, log loss, train time,
  inference time) and comparison plots in results/models/plots/
- Writes results/models/oof_probs.parquet for the chosen model (docs/CONTRACTS.md #5)
- `--final` (once, after the 15:30 freeze): fit on all train, score student_test.parquet,
  write results/models/final_student_test.csv. Never use these numbers to change anything.
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
