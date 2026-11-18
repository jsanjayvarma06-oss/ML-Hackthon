"""Dataset exploration. OWNER: Samarth.

Run from the repo root:  python scripts/eda.py
Partition used: TRAIN only (data/raw/train). student_test and evaluator trips are not read.

Writes
  results/eda/trip_inventory.csv   one row per trip (points, duration, sampling, gaps,
                                   speed spikes, non-finite rows, network size, duplicate
                                   arcs, self-loops, components, route length)
  results/eda/missing_values.csv   non-finite / bad values per file type
  results/eda/*.png                sampling histogram, network size, trajectories
  results/maps/eda_<trip>.html     a few maps made with pathsnap.viz.trip_map
The route file is read here only to describe the data (route length, unconnected steps).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

from pathsnap import io, viz  # noqa: E402
from pathsnap.config import RESULTS_DIR, TRAIN_DIR  # noqa: E402

M_PER_DEG = 111_320.0
GAP_S = 60.0       # a gap is a step longer than this many seconds
SPIKE_MS = 70.0    # implied speed above this is a spike (m/s)
MAP_TRIPS = ["00000005", "00000013", "00000078"]  # small, huge network, huge time gap


def step_metres(lonlat):
    """Distance in metres between consecutive points (local equirectangular, per trip)."""
    lat0 = np.radians(np.nanmean(lonlat[:, 1]))
    d = np.diff(lonlat, axis=0)
    return np.hypot(d[:, 0] * M_PER_DEG * np.cos(lat0), d[:, 1] * M_PER_DEG)


def arc_lengths(trip):
    """Length in metres of every arc (local equirectangular projection)."""
    lat0 = np.radians(trip.nodes[:, 1].mean())
    p, q = trip.nodes[trip.arcs[:, 0]], trip.nodes[trip.arcs[:, 1]]
    return np.hypot((q[:, 0] - p[:, 0]) * M_PER_DEG * np.cos(lat0), (q[:, 1] - p[:, 1]) * M_PER_DEG)


def inspect(trip, route):
    """One inventory row (dict) for a trip: sampling, gaps, spikes, network and route stats."""
    t, xy = trip.track[:, 2], trip.track[:, :2]
    finite = np.isfinite(trip.track).all(axis=1)
    ok = trip.track[finite]
    dt = np.diff(ok[:, 2])
    dist = step_metres(ok[:, :2])
    pos = dt > 0
    speed = dist[pos] / dt[pos]
    n_nodes, n_arcs = len(trip.nodes), len(trip.arcs)
    a = trip.arcs
    graph = coo_matrix((np.ones(n_arcs), (a[:, 0], a[:, 1])), shape=(n_nodes, n_nodes))
    n_comp = connected_components(graph, directed=False)[0]
    lens = arc_lengths(trip)
    route_ok = (route >= 0) & (route < n_arcs)
    r = route[route_ok]
    unconnected = int((a[r[:-1], 1] != a[r[1:], 0]).sum()) if len(r) > 1 else 0
    return {
        "trip_id": trip.trip_id,
        "points": len(t),
        "non_finite_rows": int((~finite).sum()),
        "duration_s": float(ok[-1, 2] - ok[0, 2]),
        "dt_median_s": float(np.median(dt)),
        "dt_mean_s": float(dt.mean()),
        "dt_max_s": float(dt.max()),
        "fractional_timestamps": bool(np.any(np.abs(ok[:, 2] - np.round(ok[:, 2])) > 1e-6)),
        "non_increasing_time_steps": int((dt <= 0).sum()),
        "gaps_over_60s": int((dt > GAP_S).sum()),
        "max_speed_ms": float(speed.max()) if len(speed) else float("nan"),
        "speed_spikes_over_70ms": int((speed > SPIKE_MS).sum()),
        "path_length_m": float(dist.sum()),
        "nodes": n_nodes,
        "arcs": n_arcs,
        "duplicate_arcs": int(n_arcs - len(np.unique(a, axis=0))),
        "self_loops": int((a[:, 0] == a[:, 1]).sum()),
        "components": int(n_comp),
        "route_arcs": len(route),
        "route_arcs_out_of_range": int((~route_ok).sum()),
        "route_unconnected_steps": unconnected,
        "route_length_m": float(lens[r].sum()),
    }


def missing_table(inv, raw_nonfinite):
    """Table of data-quality checks and how many problems each found over all train trips."""
    rows = [
        ("track rows with a non-finite value", int(inv["non_finite_rows"].sum())),
        ("track time steps with dt <= 0", int(inv["non_increasing_time_steps"].sum())),
        ("route arc ids outside the arcs file", int(inv["route_arcs_out_of_range"].sum())),
        ("nodes with a non-finite coordinate", raw_nonfinite["nodes"]),
        ("arcs referencing a missing node", raw_nonfinite["arcs"]),
    ]
    return pd.DataFrame(rows, columns=["check", "count_over_all_train_trips"])


def plots(inv, out, tracks):
    """Save sampling_hist.png, network_size.png and trajectories.png into folder `out`."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(inv["dt_median_s"], bins=20, color="#4477aa")
    ax[0].set(title="Median sampling interval per trip", xlabel="seconds", ylabel="trips")
    ax[1].hist(np.log10(inv["dt_max_s"].clip(lower=1e-3)), bins=20, color="#cc6677")
    ax[1].set(title="Largest time step per trip", xlabel="log10(seconds)", ylabel="trips")
    fig.tight_layout(); fig.savefig(out / "sampling_hist.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(inv["arcs"], inv["points"], s=14, color="#228833")
    ax.set(xscale="log", yscale="log", xlabel="arcs in network", ylabel="GPS points",
           title="Network size vs trip length (train)")
    fig.tight_layout(); fig.savefig(out / "network_size.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    for a, (tid, xy) in zip(ax.ravel(), tracks):
        a.plot(xy[:, 0], xy[:, 1], lw=0.6, color="#4477aa")
        a.set_title(f"trip {tid}", fontsize=8); a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal", "datalim")
    for a in ax.ravel()[len(tracks):]:
        a.axis("off")
    fig.suptitle("GPS trajectories (first 12 train trips; lon/lat axes)")
    fig.tight_layout(); fig.savefig(out / "trajectories.png", dpi=130); plt.close(fig)


def main():
    """Run the full EDA on train trips and write CSVs, plots and maps."""
    eda_dir = RESULTS_DIR / "eda"
    maps_dir = RESULTS_DIR / "maps"
    eda_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    ids = io.list_trips(TRAIN_DIR)
    print(f"EDA on TRAIN only: {len(ids)} trips")
    rows, tracks = [], []
    raw_bad = {"nodes": 0, "arcs": 0}
    for tid in ids:
        trip = io.load_trip(TRAIN_DIR, tid)
        route = io.load_route(TRAIN_DIR, tid)
        rows.append(inspect(trip, route))
        raw_bad["nodes"] += int((~np.isfinite(trip.nodes)).any(axis=1).sum())
        raw_bad["arcs"] += int(((trip.arcs < 0) | (trip.arcs >= len(trip.nodes))).any(axis=1).sum())
        if len(tracks) < 12:
            tracks.append((tid, trip.track[np.isfinite(trip.track).all(axis=1), :2]))
    inv = pd.DataFrame(rows)
    inv.to_csv(eda_dir / "trip_inventory.csv", index=False)
    missing_table(inv, raw_bad).to_csv(eda_dir / "missing_values.csv", index=False)
    plots(inv, eda_dir, tracks)

    for tid in MAP_TRIPS:
        if tid not in ids:
            print(f"skip map {tid}: not in train")
            continue
        trip = io.load_trip(TRAIN_DIR, tid)
        route = io.load_route(TRAIN_DIR, tid)
        viz.trip_map(trip, route=route).save(str(maps_dir / f"eda_{tid}.html"))
        print("map", tid)
    print(inv.describe().T[["min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
