"""Features for each (GPS point, candidate arc) pair. OWNER: Sanjay.

Output must follow the candidate table in docs/CONTRACTS.md.
Never use anything derived from the route file as a feature.
"""

# TODO(Sanjay): final list. Kushal's model code reads this list, so keep it current.
FEATURE_COLUMNS = [
    # e.g. "dist_m", "dist_rank", "heading_diff_deg", "arc_len_m", "proj_t", "speed_mps", ...
]


def build_candidate_table(trip, candidates, point_labels=None):
    """One row per (point, candidate) with the contract columns + FEATURE_COLUMNS.

    point_labels: output of labels.make_point_labels, or None at prediction time
    (then true_arc and label are filled with -1).
    Returns a pandas DataFrame.
    """
    raise NotImplementedError("TODO(Sanjay)")
