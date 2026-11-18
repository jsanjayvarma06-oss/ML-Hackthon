"""Turn the reference route into one label per GPS point. OWNER: Sanjay.

The .route file is an ordered list of arcs with no timestamps, and it has far fewer
rows than the track. Align them in order: walk forward along the route and give each
GPS point the nearest route arc, never jumping backwards. Points too far from the
route get metrics.UNLABELLED (-1) and are left out of training and scoring.

Known label problems to mention in the report: some routes contain arcs that do not
connect, and in some trips GPS points are hundreds of metres from the route.
"""

MAX_LABEL_DIST_M = 50.0  # TODO(Sanjay): choose and justify


def make_point_labels(trip, route, max_dist_m=MAX_LABEL_DIST_M):
    """Reference arc id for every GPS point (int array, -1 = unlabelled)."""
    raise NotImplementedError("TODO(Sanjay)")
