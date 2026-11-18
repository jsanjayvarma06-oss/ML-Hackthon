"""Cleaning, label and candidate statistics for train and student_test. OWNER: Sanjay.

Run from the repo root:  python scripts/candidate_stats.py
Writes to results/pipeline/:
    trip_stats.csv           per trip: cleaning counts, label coverage, label distance
    candidate_miss_rate.csv  miss rate for each (partition, K, radius)
K=1 with the largest radius is the "nearest arc" baseline (1 - miss rate = its Top-1).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap import candidates, io, labels  # noqa: E402
from pathsnap.config import RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR  # noqa: E402
from pathsnap.geo import project_trip  # noqa: E402
from pathsnap.metrics import UNLABELLED  # noqa: E402

KS = [1, 3, 5, 8, 12, 16]
RADII = [30, 50, 75, 100, 150]


def main():
    out = RESULTS_DIR / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    trip_rows, point_rows = [], []

    for part_name, part in [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]:
        for trip_id in io.list_trips(part):
            t0 = time.perf_counter()
            trip = io.load_trip(part, trip_id)
            route = io.load_route(part, trip_id)
            keep, seg_id, info = candidates.clean_track(trip.track)
            txy = project_trip(trip)
            _, route_arc, dist = labels.align_route(trip, route, keep, txy)
            true_arc = labels.make_point_labels(trip, route, keep_idx=keep, txy=txy)
            arcs, dists = candidates.generate_candidates(trip, k=max(KS), radius_m=max(RADII),
                                                         keep_idx=keep, txy=txy)
            secs = time.perf_counter() - t0

            for row in np.flatnonzero(true_arc != UNLABELLED):
                a, d = arcs[row], dists[row]
                hit = np.flatnonzero(a == true_arc[row])
                point_rows.append((part_name, hit[0] if len(hit) else -1,
                                   d[hit[0]] if len(hit) else np.inf))

            trip_rows.append({
                "partition": part_name, "trip_id": trip_id, **info,
                "n_labelled": int((true_arc != UNLABELLED).sum()),
                "label_coverage": float((true_arc != UNLABELLED).sum() / max(1, info["n_kept"])),
                "label_dist_median_m": float(np.median(dist)),
                "label_dist_p95_m": float(np.percentile(dist, 95)),
                "n_arcs": int(len(trip.arcs)),
                "seconds": round(secs, 2),
            })
            print(f"{part_name}/{trip_id}: kept {info['n_kept']}/{info['n_raw']}, "
                  f"labelled {trip_rows[-1]['label_coverage']:.1%}, {secs:.1f}s", flush=True)

    trips = pd.DataFrame(trip_rows)
    trips.to_csv(out / "trip_stats.csv", index=False)

    pts = pd.DataFrame(point_rows, columns=["partition", "true_rank", "true_dist"])
    rows = []
    for part_name, g in pts.groupby("partition"):
        for k in KS:
            for r in RADII:
                found = (g["true_rank"] >= 0) & (g["true_rank"] < k) & (g["true_dist"] <= r)
                rows.append({"partition": part_name, "k": k, "radius_m": r,
                             "n_points": len(g), "miss_rate": float(1 - found.mean())})
    miss = pd.DataFrame(rows)
    miss.to_csv(out / "candidate_miss_rate.csv", index=False)

    print("\nLabelled points:", trips.groupby("partition")["n_labelled"].sum().to_dict())
    print("\nMiss rate (rows: K, columns: radius m)")
    for part_name, g in miss.groupby("partition"):
        print(f"\n{part_name}")
        print(g.pivot(index="k", columns="radius_m", values="miss_rate").round(4).to_string())


if __name__ == "__main__":
    main()
