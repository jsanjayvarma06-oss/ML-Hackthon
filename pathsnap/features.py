"""Features for each (GPS point, candidate arc) pair. OWNER: Sanjay.

Output must follow the candidate table in docs/CONTRACTS.md.
Never use anything derived from the route file as a feature.
"""
import numpy as np
import pandas as pd

from pathsnap.candidates import clean_track, generate_candidates
from pathsnap.geo import angle_diff_deg, bearing_deg, point_to_segment, project_trip
from pathsnap.metrics import UNLABELLED

FEATURE_COLUMNS = [
    "dist_m",
    "heading_diff_deg",
    "arc_len_m",
    "proj_t",
    "speed_mps",
    "dist_to_start_m",
    "dist_to_end_m",
]


def extract_features(trip, cand_arcs, cand_dist, keep_idx=None, txy=None):
    """Extract features for each (point, candidate) pair.

    Returns:
        rows: list of dicts or DataFrame containing metadata and FEATURE_COLUMNS.
    """
    if keep_idx is None:
        keep_idx, _, _ = clean_track(trip.track)
    if txy is None:
        txy = project_trip(trip)

    # Compute GPS movement headings and speeds
    pts = txy.track_xy[keep_idx]
    times = trip.track[keep_idx, 2]
    n_kept = len(keep_idx)

    gps_bearings = np.zeros(n_kept, dtype=float)
    gps_speeds = np.zeros(n_kept, dtype=float)

    if n_kept > 1:
        dx = np.diff(pts[:, 0])
        dy = np.diff(pts[:, 1])
        dt = np.diff(times)
        dt_safe = np.where(dt > 0, dt, 1.0)
        step_dist = np.hypot(dx, dy)
        step_speeds = step_dist / dt_safe
        step_bearings = bearing_deg(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1])

        # Point heading is average of incoming and outgoing when available
        gps_bearings[:-1] = step_bearings
        gps_bearings[-1] = step_bearings[-1] if len(step_bearings) > 0 else 0.0

        gps_speeds[:-1] = step_speeds
        gps_speeds[-1] = step_speeds[-1] if len(step_speeds) > 0 else 0.0

    # Build feature dicts
    records = []
    seg = txy.seg
    arc_len = txy.arc_len
    arc_bearing = txy.arc_bearing

    for i, row_idx in enumerate(keep_idx):
        arcs = cand_arcs[row_idx]
        dists = cand_dist[row_idx]
        if len(arcs) == 0:
            continue

        px, py = pts[i]
        g_bearing = gps_bearings[i]
        speed = gps_speeds[i]

        s = seg[arcs]
        _, proj_t = point_to_segment(px, py, s[:, 0], s[:, 1], s[:, 2], s[:, 3])
        h_diff = angle_diff_deg(g_bearing, arc_bearing[arcs])
        d_start = np.hypot(px - s[:, 0], py - s[:, 1])
        d_end = np.hypot(px - s[:, 2], py - s[:, 3])

        for rank, arc_id in enumerate(arcs):
            records.append({
                "point_idx": int(row_idx),
                "cand_rank": int(rank),
                "arc_id": int(arc_id),
                "dist_m": float(dists[rank]),
                "heading_diff_deg": float(h_diff[rank]),
                "arc_len_m": float(arc_len[arc_id]),
                "proj_t": float(proj_t[rank]),
                "speed_mps": float(speed),
                "dist_to_start_m": float(d_start[rank]),
                "dist_to_end_m": float(d_end[rank]),
            })

    return records


def build_candidate_table(trip, candidates_tuple, point_labels=None):
    """One row per (point, candidate) with the contract columns + FEATURE_COLUMNS.

    candidates_tuple: (cand_arcs, cand_dist) from candidates.generate_candidates
    point_labels: output of labels.make_point_labels, or None at prediction time
    (then true_arc and label are filled with -1).
    Returns a pandas DataFrame.
    """
    cand_arcs, cand_dist = candidates_tuple
    records = extract_features(trip, cand_arcs, cand_dist)

    if not records:
        cols = ["trip_id", "point_idx", "cand_rank", "arc_id", "true_arc", "label"] + FEATURE_COLUMNS
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame.from_records(records)
    df["trip_id"] = str(trip.trip_id)

    if point_labels is not None:
        point_labels = np.asarray(point_labels)
        df["true_arc"] = point_labels[df["point_idx"].values]
        df["label"] = ((df["arc_id"] == df["true_arc"]) & (df["true_arc"] != UNLABELLED)).astype(int)
    else:
        df["true_arc"] = UNLABELLED
        df["label"] = -1

    ordered_cols = ["trip_id", "point_idx", "cand_rank", "arc_id", "true_arc", "label"] + FEATURE_COLUMNS
    return df[ordered_cols]
