"""Build candidate + feature tables for train and student_test. OWNER: Sanjay.

Run from the repo root:  python scripts/build_dataset.py
Writes data/processed/train.parquet and data/processed/student_test.parquet
in the format from docs/CONTRACTS.md, plus results/pipeline/dataset_summary.csv
(rows, points, labelled points, candidate miss rate per trip).

Labels come from the route file through labels.make_point_labels. The route is used
only for true_arc / label, never for a feature.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap import features, io, labels, metrics  # noqa: E402
from pathsnap.config import PROCESSED_DIR, RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR  # noqa: E402


def one_trip(args):
    part_dir, trip_id = args
    t0 = time.perf_counter()
    trip = io.load_trip(part_dir, trip_id)
    route = io.load_route(part_dir, trip_id)
    from pathsnap.candidates import clean_track
    from pathsnap.geo import project_trip
    keep_idx, _, _ = clean_track(trip.track)
    txy = project_trip(trip)
    point_labels = labels.make_point_labels(trip, route, keep_idx=keep_idx, txy=txy)
    df, cand_arcs, _, keep_idx, _, _ = features.trip_table(trip, point_labels)
    miss = metrics.candidate_miss_rate(point_labels, cand_arcs)
    summary = {
        "trip_id": trip_id,
        "n_points": len(trip.track),
        "n_kept": len(keep_idx),
        "n_labelled": int((point_labels != metrics.UNLABELLED).sum()),
        "n_rows": len(df),
        "candidate_miss_rate": miss,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    return df, summary


def build(part_dir, name, workers):
    ids = io.list_trips(part_dir)
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(one_trip, [(part_dir, t) for t in ids]))
    df = pd.concat([d for d, _ in out], ignore_index=True)
    summ = pd.DataFrame([s for _, s in out])
    summ.insert(0, "partition", name)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    lab = df[df.true_arc != metrics.UNLABELLED]
    print(f"{name}: {len(ids)} trips, {len(df):,} rows, {df.groupby(['trip_id','point_idx']).ngroups:,} points, "
          f"positive rate among labelled rows {lab.label.mean():.4f}")
    return summ


def main():
    workers = min(8, max(1, (len(__import__('os').sched_getaffinity(0)))))
    summ = [build(TRAIN_DIR, "train", workers)]
    if "--train-only" not in sys.argv:
        summ.append(build(STUDENT_TEST_DIR, "student_test", workers))
    out = RESULTS_DIR / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    s = pd.concat(summ, ignore_index=True)
    s.to_csv(out / "dataset_summary.csv", index=False)
    for part, g in s.groupby("partition"):
        lab = g.n_labelled.sum()
        miss = np.nansum(g.candidate_miss_rate * g.n_labelled) / lab
        print(f"{part}: labelled points {lab:,}, pooled candidate miss rate {miss:.4f}")


if __name__ == "__main__":
    main()
