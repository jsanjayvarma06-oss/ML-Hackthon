"""Build candidate + feature tables for train and student_test. OWNER: Sanjay.

Run from the repo root:  python scripts/build_dataset.py
Writes data/processed/train.parquet and data/processed/student_test.parquet
in the format from docs/CONTRACTS.md, plus results/pipeline/candidate_miss_rate.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathsnap import candidates, features, io, labels  # noqa: E402
from pathsnap.config import PROCESSED_DIR, STUDENT_TEST_DIR, TRAIN_DIR  # noqa: E402


def main():
    raise NotImplementedError("TODO(Sanjay)")


if __name__ == "__main__":
    main()
