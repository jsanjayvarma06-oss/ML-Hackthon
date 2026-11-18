"""Track cleaning and candidate road segments for each GPS point. OWNER: Sanjay.

All outputs are indexed by the row number in the original .track file, so labels,
features and predictions line up with the raw GPS points. Points removed by cleaning
simply get no candidates.

Some trips have 300k arcs, so candidates come from a KD-tree over points sampled
every SAMPLE_STEP_M metres along each arc, then exact point-to-segment distances.
"""
import numpy as np
from scipy.spatial import cKDTree

from pathsnap.geo import point_to_segment, project_trip, to_local_xy

# Chosen on train only (results/pipeline/candidate_miss_rate.csv): K=8 misses 0.9% of
# labelled train points; K=16 only saves another 0.4% but doubles the rows.
# K=1 misses ~45% because two-way roads give two arcs at the same distance.
K_CANDIDATES = 8
RADIUS_M = 75.0           # above the 50 m label cut-off, so a labelled arc is never cut by radius
MAX_SPEED_MPS = 70.0      # 252 km/h: anything faster between two fixes is a GPS spike
MAX_SPIKE_RUN = 5         # after this many drops in a row, trust the track again
GAP_SPLIT_S = 60.0        # a time gap longer than this starts a new segment
SAMPLE_STEP_M = 10.0


def clean_track(track):
    """Pick which GPS rows to keep and split the trip at big time gaps.

    Steps: drop non-finite rows, sort by time, drop repeated or backward timestamps,
    drop fixes that would need a speed above MAX_SPEED_MPS from the last kept fix.

    Returns (keep_idx, segment_id, info):
        keep_idx    int array of kept row numbers in the original track, in time order
        segment_id  int array, same length, increases after every gap > GAP_SPLIT_S
        info        dict of counts for the report
    """
    finite = np.flatnonzero(np.isfinite(track).all(axis=1))
    order = finite[np.argsort(track[finite, 2], kind="stable")]
    x, y = to_local_xy(track[:, 0], track[:, 1], track[order, 0].mean(), track[order, 1].mean())

    keep, n_time, n_spike, run = [order[0]], 0, 0, 0
    for i in order[1:]:
        j = keep[-1]
        dt = track[i, 2] - track[j, 2]
        if dt <= 0:
            n_time += 1
            continue
        if np.hypot(x[i] - x[j], y[i] - y[j]) / dt > MAX_SPEED_MPS and run < MAX_SPIKE_RUN:
            n_spike += 1
            run += 1
            continue
        keep.append(i)
        run = 0
    keep = np.asarray(keep)

    gaps = np.diff(track[keep, 2]) > GAP_SPLIT_S
    segment_id = np.concatenate([[0], np.cumsum(gaps)])
    info = {
        "n_raw": int(len(track)),
        "n_non_finite": int(len(track) - len(finite)),
        "n_bad_time": n_time,
        "n_speed_spike": n_spike,
        "n_kept": int(len(keep)),
        "n_segments": int(segment_id[-1] + 1),
    }
    return keep, segment_id, info


def canonical_arc_ids(arcs):
    """Map every arc to the first arc with the same (from, to) nodes.

    Some networks contain exact duplicate arcs; treating them as one road avoids a
    coin flip between identical candidates.
    """
    _, first, inverse = np.unique(arcs, axis=0, return_index=True, return_inverse=True)
    return first[inverse.ravel()]


def usable_arcs(arcs):
    """Arc ids that can be candidates: not a self-loop, first copy of any duplicate."""
    canon = canonical_arc_ids(arcs)
    ids = np.arange(len(arcs))
    return ids[(canon == ids) & (arcs[:, 0] != arcs[:, 1])]


def generate_candidates(trip, k=K_CANDIDATES, radius_m=RADIUS_M, keep_idx=None, txy=None):
    """For every GPS point, the k nearest usable arcs within radius_m.

    keep_idx: rows to use (from clean_track); computed if not given.
    txy: geo.project_trip(trip); computed if not given.
    Returns (cand_arcs, cand_dist): two lists with one entry per row of the original
    track. Each entry is an array sorted nearest first; empty if the point was
    removed by cleaning or has no arc within radius_m.
    """
    if keep_idx is None:
        keep_idx = clean_track(trip.track)[0]
    if txy is None:
        txy = project_trip(trip)
    n = len(trip.track)
    empty_i, empty_f = np.empty(0, dtype=int), np.empty(0)
    cand_arcs, cand_dist = [empty_i] * n, [empty_f] * n

    # Only arcs whose bounding box comes near the track.
    pts = txy.track_xy[keep_idx]
    lo, hi = pts.min(axis=0) - radius_m, pts.max(axis=0) + radius_m
    arc_ids = usable_arcs(trip.arcs)
    s = txy.seg[arc_ids]
    near = ((np.maximum(s[:, 0], s[:, 2]) >= lo[0]) & (np.minimum(s[:, 0], s[:, 2]) <= hi[0])
            & (np.maximum(s[:, 1], s[:, 3]) >= lo[1]) & (np.minimum(s[:, 1], s[:, 3]) <= hi[1]))
    arc_ids, s = arc_ids[near], s[near]
    if len(arc_ids) == 0:
        return cand_arcs, cand_dist

    # Sample points every SAMPLE_STEP_M along each arc (both ends included).
    n_samp = np.ceil(txy.arc_len[arc_ids] / SAMPLE_STEP_M).astype(int) + 1
    owner = np.repeat(np.arange(len(arc_ids)), n_samp)
    frac = np.concatenate([np.linspace(0, 1, m) for m in n_samp])
    sx = s[owner, 0] + frac * (s[owner, 2] - s[owner, 0])
    sy = s[owner, 1] + frac * (s[owner, 3] - s[owner, 1])
    tree = cKDTree(np.column_stack([sx, sy]))

    # Any arc within radius_m has a sample within radius_m + half a step.
    hits = tree.query_ball_point(pts, r=radius_m + SAMPLE_STEP_M / 2)
    for row, (px, py), h in zip(keep_idx, pts, hits):
        if not h:
            continue
        local = np.unique(owner[h])
        seg = s[local]
        d, _ = point_to_segment(px, py, seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3])
        ok = d <= radius_m
        if not ok.any():
            continue
        local, d = local[ok], d[ok]
        best = np.argsort(d, kind="stable")[:k]
        cand_arcs[row] = arc_ids[local[best]]
        cand_dist[row] = d[best]
    return cand_arcs, cand_dist
