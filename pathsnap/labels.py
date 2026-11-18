"""Turn the reference route into one label per GPS point. OWNER: Sanjay.

The .route file is an ordered list of arcs with no timestamps, and it has far fewer
rows than the track. We align them with dynamic programming: give every kept GPS point
(in time order) a position in the route so that positions never go backwards and the
total point-to-arc distance is as small as possible. Positions may skip arcs (short
arcs often get no GPS point). Points still too far from their arc get UNLABELLED (-1)
and are left out of training and scoring.

Known label problems to mention in the report: some routes contain arcs that do not
connect, and in some trips GPS points are hundreds of metres from the route.
"""
import numpy as np

from pathsnap.candidates import canonical_arc_ids, clean_track
from pathsnap.geo import point_to_segment, project_trip
from pathsnap.metrics import UNLABELLED

# Aligned points sit a median ~4.7 m and p95 ~15 m from their route arc
# (results/pipeline/trip_stats.csv), so 50 m keeps real GPS noise and drops points
# the route does not cover (trip 54: route covers only the first ~11% of the track).
MAX_LABEL_DIST_M = 50.0
COST_CAP_M = 100.0       # far-off points should not drag the whole alignment


def align_route(trip, route, keep_idx=None, txy=None):
    """Monotone alignment of kept GPS points to route positions.

    Returns (keep_idx, route_arc, dist): for each kept row, the canonical arc id it is
    aligned to and the distance to that arc in metres.
    """
    if keep_idx is None:
        keep_idx = clean_track(trip.track)[0]
    if txy is None:
        txy = project_trip(trip)
    canon = canonical_arc_ids(trip.arcs)
    route = np.asarray(route)
    route = canon[route[(route >= 0) & (route < len(trip.arcs))]]
    seg = txy.seg[route]
    ax, ay, bx, by = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
    pts = txy.track_xy[keep_idx]
    n, r = len(pts), len(route)
    idx = np.arange(r)

    # total[j] = best cost so far with the current point on route position j
    back = np.empty((n, r), dtype=np.int32)
    total = None
    for i, (px, py) in enumerate(pts):
        cost = np.minimum(point_to_segment(px, py, ax, ay, bx, by)[0], COST_CAP_M)
        if total is None:
            total = cost
            back[i] = idx
            continue
        prefix_min = np.minimum.accumulate(total)
        # position of that prefix minimum (last index where a new minimum was reached)
        back[i] = np.maximum.accumulate(np.where(total <= prefix_min, idx, 0))
        total = cost + prefix_min

    pos = np.empty(n, dtype=int)
    pos[-1] = int(np.argmin(total))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    dist, _ = point_to_segment(pts[:, 0], pts[:, 1], ax[pos], ay[pos], bx[pos], by[pos])
    return keep_idx, route[pos], dist


def make_point_labels(trip, route, max_dist_m=MAX_LABEL_DIST_M, keep_idx=None, txy=None):
    """Reference arc id for every row of the original track (int array, -1 = unlabelled).

    Arc ids are canonical (see candidates.canonical_arc_ids), matching the candidates.
    """
    keep_idx, route_arc, dist = align_route(trip, route, keep_idx, txy)
    labels = np.full(len(trip.track), UNLABELLED, dtype=int)
    ok = dist <= max_dist_m
    labels[keep_idx[ok]] = route_arc[ok]
    return labels
