"""Build candidate + feature tables. OWNER: Sanjay.

Run from the repo root:
    python scripts/build_dataset.py            # train only -> data/processed/train.parquet
    python scripts/build_dataset.py --final    # also student_test (only after the 15:30 freeze)
Format: docs/CONTRACTS.md #1.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from pathsnap import candidates, features, io, labels  # noqa: E402
from pathsnap.config import PROCESSED_DIR, STUDENT_TEST_DIR, TRAIN_DIR  # noqa: E402
from pathsnap.geo import project_trip  # noqa: E402


def build_partition(part_dir):
    tables = []
    for trip_id in io.list_trips(part_dir):
        t0 = time.perf_counter()
        trip = io.load_trip(part_dir, trip_id)
        keep, seg_id, _ = candidates.clean_track(trip.track)
        txy = project_trip(trip)
        point_labels = labels.make_point_labels(trip, io.load_route(part_dir, trip_id),
                                                keep_idx=keep, txy=txy)
        arcs, dists = candidates.generate_candidates(trip, keep_idx=keep, txy=txy)
        tab = features.build_candidate_table(trip, arcs, dists, keep, seg_id, txy, point_labels)
        tables.append(tab)
        print(f"  {trip_id}: {len(tab):,} rows, {time.perf_counter() - t0:.1f}s", flush=True)
    return pd.concat(tables, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="also build student_test (after freeze)")
    args = ap.parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    parts = [("train", TRAIN_DIR)] + ([("student_test", STUDENT_TEST_DIR)] if args.final else [])
    for name, part_dir in parts:
        print(f"{name}:")
        df = build_partition(part_dir)
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        lab = df[df["true_arc"] != -1]
        print(f"{name}: {len(df):,} rows, {df['trip_id'].nunique()} trips, "
              f"{lab.groupby(['trip_id', 'point_idx']).ngroups:,} labelled points, "
              f"positive rate {lab['label'].mean():.3f}")


if __name__ == "__main__":
    main()
