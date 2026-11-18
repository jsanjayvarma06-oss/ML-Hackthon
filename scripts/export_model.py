"""Refit the tuned model on all train data and save the bundle. OWNER: Kushal.

Run from the repo root:  python scripts/export_model.py
Writes artifacts/model.joblib in the format from docs/CONTRACTS.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathsnap.config import ARTIFACTS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED  # noqa: E402


def main():
    raise NotImplementedError("TODO(Kushal)")


if __name__ == "__main__":
    main()
