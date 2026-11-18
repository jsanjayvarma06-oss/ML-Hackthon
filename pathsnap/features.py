"""Features for each (GPS point, candidate arc) pair. OWNER: Sanjay.

Output follows the candidate table in docs/CONTRACTS.md.
Everything here is computed from the track, nodes and arcs only. The route file is
never read here; labels are passed in from labels.py and only fill true_arc / label.

The same function builds the training table and the live table inside the API, so the
model always sees features computed the same way (no train/serve skew).
"""
import numpy as np
import pandas as pd

from pathsnap.candidates import clean_track, generate_candidates
from pathsnap.geo import angle_diff_deg, bearing_deg, point_to_segment, project_trip
from pathsnap.metrics import UNLABELLED

# A GPS heading is only meaningful if the car moved this far between the
# neighbouring fixes; when it is (nearly) stopped the heading is pure noise.
MIN_MOVE_FOR_HEADING_M = 2.0

FEATURE_COLUMNS = [
    "dist_m",             # GPS point to arc, metres
    "dist_rank",          # 0 = nearest candidate
    "dist_gap_to_best",   # dist_m minus the nearest candidate's distance
    "heading_diff_deg",   # GPS travel direction vs arc direction, 0..180
    "heading_reliable",   # 1 if the car moved enough for the heading to mean something
    "arc_len_m",          # arc length
    "proj_t",             # where the point projects along the arc, 0..1
    "speed_mps",          # GPS speed from the neighbouring fixes
    "n_candidates",       # candidates for this point (dense vs sparse area)
    "has_reverse_twin",   # 1 if the reverse arc (to -> from) is also a candidate
    "prev_dist_m",        # previous kept GPS point to this same arc
    "next_dist_m",        # next kept GPS point to this same arc
    "conn_prev_nearest",  # 1 if this arc equals / continues the previous point's nearest arc
]


def _motion(txy, track, keep_idx, segment_id):
    """Heading (deg), movement (m) and speed (m/s) per kept point.

    Uses the previous and next kept point inside the same segment (central
    difference); at segment ends it falls back to a one-sided difference.
    """
    p = txy.track_xy[keep_idx]
    t = track[keep_idx, 2]
    n = len(keep_idx)
    prev = np.arange(n) - 1
    nxt = np.arange(n) + 1
    prev[0] = 0
    nxt[-1] = n - 1
    seg_prev_ok = np.r_[False, segment_id[1:] == segment_id[:-1]]
    seg_next_ok = np.r_[segment_id[:-1] == segment_id[1:], False]
    prev = np.where(seg_prev_ok, prev, np.arange(n))
    nxt = np.where(seg_next_ok, nxt, np.arange(n))
    dx = p[nxt, 0] - p[prev, 0]
    dy = p[nxt, 1] - p[prev, 1]
    move = np.hypot(dx, dy)
    dt = t[nxt] - t[prev]
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(dt > 0, move / dt, 0.0)
    heading = bearing_deg(p[prev, 0], p[prev, 1], p[nxt, 0], p[nxt, 1])
    return heading, move, speed, prev, nxt


def build_candidate_table(trip, candidates, point_labels=None, keep_idx=None,
                          segment_id=None, txy=None):
    """One row per (point, candidate) with the contract columns + FEATURE_COLUMNS.

    candidates: (cand_arcs, cand_dist) from candidates.generate_candidates.
    point_labels: output of labels.make_point_labels, or None at prediction time
    (then true_arc and label are -1 / 0).
    keep_idx, segment_id: from candidates.clean_track (computed if not given).
    Returns a pandas DataFrame (empty if the trip has no candidates).
    """
    cand_arcs, cand_dist = candidates
    if keep_idx is None or segment_id is None:
        keep_idx, segment_id, _ = clean_track(trip.track)
    if txy is None:
        txy = project_trip(trip)

    heading, move, speed, prev, nxt = _motion(txy, trip.track, keep_idx, segment_id)
    nearest = np.array([c[0] if len(c) else -1 for c in (cand_arcs[r] for r in keep_idx)])

    rows_k, ranks, arcs, dists = [], [], [], []
    for k, r in enumerate(keep_idx):
        c = cand_arcs[r]
        if len(c) == 0:
            continue
        rows_k.append(np.full(len(c), k))
        ranks.append(np.arange(len(c)))
        arcs.append(c)
        dists.append(cand_dist[r])
    if not arcs:
        return pd.DataFrame(columns=["trip_id", "point_idx", "cand_rank", "arc_id",
                                     "true_arc", "label", *FEATURE_COLUMNS])
    k = np.concatenate(rows_k)
    rank = np.concatenate(ranks)
    arc = np.concatenate(arcs).astype(int)
    dist = np.concatenate(dists)
    row = keep_idx[k]

    # distance to the nearest candidate of the same point
    best = np.zeros(len(keep_idx))
    first = np.r_[True, k[1:] != k[:-1]]
    best[k[first]] = dist[first]
    n_cand = np.bincount(k, minlength=len(keep_idx))

    seg = txy.seg[arc]
    px, py = txy.track_xy[row, 0], txy.track_xy[row, 1]
    _, proj_t = point_to_segment(px, py, seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3])

    def dist_from(other_k):
        q = txy.track_xy[keep_idx[other_k]]
        d, _ = point_to_segment(q[:, 0], q[:, 1], seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3])
        return d

    # reverse twin among this point's candidates
    frm, to = trip.arcs[arc, 0], trip.arcs[arc, 1]
    key = pd.MultiIndex.from_arrays([k, frm, to])
    rev = pd.MultiIndex.from_arrays([k, to, frm])
    has_twin = rev.isin(key).astype(int)

    # connectivity with the previous kept point's nearest arc (no label used)
    pn = nearest[prev[k]]
    has_prev = prev[k] != k
    conn = has_prev & (pn >= 0) & ((pn == arc) | (trip.arcs[np.maximum(pn, 0), 1] == frm))

    if point_labels is None:
        true_arc = np.full(len(arc), UNLABELLED)
    else:
        true_arc = np.asarray(point_labels)[row]
    label = ((true_arc == arc) & (true_arc != UNLABELLED)).astype(int)

    df = pd.DataFrame({
        "trip_id": trip.trip_id,
        "point_idx": row.astype(int),
        "cand_rank": rank.astype(int),
        "arc_id": arc,
        "true_arc": true_arc.astype(int),
        "label": label,
        "dist_m": dist,
        "dist_rank": rank.astype(float),
        "dist_gap_to_best": dist - best[k],
        "heading_diff_deg": angle_diff_deg(heading[k], txy.arc_bearing[arc]),
        "heading_reliable": (move[k] >= MIN_MOVE_FOR_HEADING_M).astype(float),
        "arc_len_m": txy.arc_len[arc],
        "proj_t": proj_t,
        "speed_mps": speed[k],
        "n_candidates": n_cand[k].astype(float),
        "has_reverse_twin": has_twin.astype(float),
        "prev_dist_m": dist_from(prev[k]),
        "next_dist_m": dist_from(nxt[k]),
        "conn_prev_nearest": conn.astype(float),
    })
    return df


def trip_table(trip, point_labels=None):
    """clean -> candidates -> features for one trip. Used by build_dataset and the pipeline.

    Returns (df, cand_arcs, cand_dist, keep_idx, segment_id, txy).
    """
    keep_idx, segment_id, _ = clean_track(trip.track)
    txy = project_trip(trip)
    cand_arcs, cand_dist = generate_candidates(trip, keep_idx=keep_idx, txy=txy)
    df = build_candidate_table(trip, (cand_arcs, cand_dist), point_labels,
                               keep_idx=keep_idx, segment_id=segment_id, txy=txy)
    return df, cand_arcs, cand_dist, keep_idx, segment_id, txy
