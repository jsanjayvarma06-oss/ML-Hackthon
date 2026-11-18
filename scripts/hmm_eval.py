"""Compare classifier-only vs classifier + HMM on student_test. OWNER: Sanjay.

Run from the repo root:  python scripts/hmm_eval.py
Writes results/pipeline/hmm_vs_classifier.csv (per trip: top1 both ways, continuity both ways)
and results/pipeline/maps/hmm_<trip>.html for a few trips.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathsnap import io, labels, metrics, pipeline  # noqa: E402
from pathsnap.config import RESULTS_DIR, STUDENT_TEST_DIR  # noqa: E402


def main():
    raise NotImplementedError("TODO(Sanjay)")


if __name__ == "__main__":
    main()
