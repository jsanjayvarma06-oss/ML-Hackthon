"""Dataset exploration. OWNER: Samarth.

Run from the repo root:  python scripts/eda.py   (or use notebooks/01_eda.ipynb)
Writes results/eda/trip_inventory.csv (per trip: points, duration, sampling,
gaps, speed spikes, nodes, arcs, components, route length), plots, and
results/maps/eda_<trip>.html. Say which partition each analysis uses.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathsnap import io  # noqa: E402
from pathsnap.config import RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR  # noqa: E402


def main():
    raise NotImplementedError("TODO(Samarth)")


if __name__ == "__main__":
    main()
