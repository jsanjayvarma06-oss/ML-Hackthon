"""Compare classifier-only vs classifier + HMM. OWNER: Sanjay.

Run from the repo root:  python scripts/hmm_eval.py
Default: train only, using Kushal's out-of-fold probabilities (docs/CONTRACTS.md #5).
Writes results/pipeline/hmm_vs_classifier.csv (per trip: top1 both ways, continuity both
ways) and is where HMM settings are tuned.
`--final` (once, after the 15:30 freeze): run the exported pipeline on student_test and write
results/pipeline/final_hmm_vs_classifier.csv. Never use these numbers to change anything.
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
