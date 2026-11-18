"""Shared paths and constants. OWNER: shared (change only after telling the team)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = RAW_DIR / "train"
STUDENT_TEST_DIR = RAW_DIR / "student_test"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"

# Folder the API serves trips from. On demo day point it at the evaluator trips:
#   export PATHSNAP_EVAL_DIR=/path/to/evaluator
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", DATA_DIR / "evaluator"))

SEED = 42
