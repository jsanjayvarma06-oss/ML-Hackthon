"""Features for each (GPS point, candidate arc) pair. OWNER: Sanjay.

Output follows the candidate table in docs/CONTRACTS.md.
Nothing here reads the route file: labels are only attached at the end, as answers.
"""
import numpy as np
import pandas as pd

from pathsnap.geo import angle_diff_deg, bearing_deg, point_to_segment
from pathsnap.metrics import UNLABELLED

MIN_MOVE_M = 3.0  # below this the GPS heading is just noise

FEATURE_COLUMNS = [
    "dist_m",            # perpendicular distance from the GPS point to the arc
    "dist_rank",         # 0 = nearest candidate of this point
    "dist_gap_to_best",  # dist_m minus the nearest candidate's dist_m
    "heading_diff_deg",  # angle between GPS direction of travel and arc direction (0..180)
    "heading_valid",     # 1 if the vehicle moved enough for a reliable heading
    "proj_t",            # where along the arc the point projects (0 = start, 1 = end)
    "arc_len_m",
    "speed_mps",         # GPS speed around this point
    "n_candidates",      # how many arcs are within the candidate radius (road density)
    "has_reverse_twin",  # 1 if the same road exists in the opposite direction
    "dist_prev_m",       # distance from the previous GPS point to this arc
    "dist_next_m",       # distance from the next GPS point to this arc
]


def point_motion(trip, keep_idx, segment_id, txy):
    """Heading (deg), heading_valid and speed (m/s) for each kept point.

    Uses the previous and next kept points in the same segment (central difference).
    Returns three arrays indexed like keep_idx, plus prev/next row numbers (-1 = none).
    """
    xy = txy.track_xy[keep_idx]
    t = trip.track[keep_idx, 2]
    n = len(keep_idx)
    prev = np.arange(n) - 1
    nxt = np.arange(n) + 1
    prev[0], nxt[-1] = 0, n - 1
    prev = np.where(segment_id[prev] == segment_id, prev, np.arange(n))
    nxt = np.where(segment_id[nxt] == segment_id, nxt, np.arange(n))

    dx, dy = xy[nxt, 0] - xy[prev, 0], xy[nxt, 1] - xy[prev, 1]
    moved = np.hypot(dx, dy)
    dt = t[nxt] - t[prev]
    heading = bearing_deg(xy[prev, 0], xy[prev, 1], xy[nxt, 0], xy[nxt, 1])
    speed = np.divide(moved, dt, out=np.zeros(n), where=dt > 0)
    prev_row = np.where(prev != np.arange(n), keep_idx[prev], -1)
    next_row = np.where(nxt != np.arange(n), keep_idx[nxt], -1)
    return heading, (moved >= MIN_MOVE_M).astype(int), speed, prev_row, next_row


def build_candidate_table(trip, cand_arcs, cand_dist, keep_idx, segment_id, txy, point_labels=None):
    """One row per (point, candidate) with the contract columns + FEATURE_COLUMNS.

    cand_arcs, cand_dist: from candidates.generate_candidates.
    keep_idx, segment_id: from candidates.clean_track. txy: geo.project_trip(trip).
    point_labels: from labels.make_point_labels, or None at prediction time
    (then true_arc and label are -1).
    """
    heading, h_valid, speed, prev_row, next_row = point_motion(trip, keep_idx, segment_id, txy)
    arc_set = {(int(u), int(v)) for u, v in trip.arcs}
    rows = []
    for i, row in enumerate(keep_idx):
        arcs, d = cand_arcs[row], cand_dist[row]
        if len(arcs) == 0:
            continue
        seg = txy.seg[arcs]
        px, py = txy.track_xy[row]
        _, proj_t = point_to_segment(px, py, seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3])
        hd = angle_diff_deg(heading[i], txy.arc_bearing[arcs]) if h_valid[i] else np.full(len(arcs), 90.0)

        def dist_from(other_row):
            if other_row < 0:
                return d  # no neighbour: fall back to this point's own distance
            ox, oy = txy.track_xy[other_row]
            return point_to_segment(ox, oy, seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3])[0]

        true_arc = UNLABELLED if point_labels is None else int(point_labels[row])
        rows.append(pd.DataFrame({
            "trip_id": trip.trip_id,
            "point_idx": int(row),
            "cand_rank": np.arange(len(arcs)),
            "arc_id": arcs,
            "true_arc": true_arc,
            "label": (arcs == true_arc).astype(int) if true_arc != UNLABELLED else UNLABELLED,
            "dist_m": d,
            "dist_rank": np.arange(len(arcs)),
            "dist_gap_to_best": d - d[0],
            "heading_diff_deg": hd,
            "heading_valid": h_valid[i],
            "proj_t": proj_t,
            "arc_len_m": txy.arc_len[arcs],
            "speed_mps": speed[i],
            "n_candidates": len(arcs),
            "has_reverse_twin": [int((int(v), int(u)) in arc_set) for u, v in trip.arcs[arcs]],
            "dist_prev_m": dist_from(prev_row[i]),
            "dist_next_m": dist_from(next_row[i]),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
