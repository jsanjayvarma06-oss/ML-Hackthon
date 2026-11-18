"""Candidate road segments for each GPS point. OWNER: Sanjay.

Some trips have 300k arcs, so use a spatial index (scipy cKDTree / shapely STRtree),
not brute force. Report the candidate miss rate with metrics.candidate_miss_rate.
"""

K_CANDIDATES = 8   # TODO(Sanjay): choose and justify
RADIUS_M = 75.0    # TODO(Sanjay): choose and justify


def clean_track(track):
    """Sort by time, drop exact duplicates, flag impossible speed spikes and big gaps.

    Returns (clean_track, info) where info records what was removed and where the
    trip should be split into segments.
    """
    raise NotImplementedError("TODO(Sanjay)")


def generate_candidates(trip, k=K_CANDIDATES, radius_m=RADIUS_M):
    """For every GPS point, the k nearest arcs within radius_m.

    Returns a list (one entry per GPS point) of int arrays of arc ids, nearest first.
    An empty array means no candidate.
    """
    raise NotImplementedError("TODO(Sanjay)")
