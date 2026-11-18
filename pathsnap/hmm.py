"""HMM + Viterbi route reconstruction. OWNER: Sanjay.

Hidden state at GPS point i = one of its candidate arcs.
Emission  = classifier probability for that candidate.
Transition = high if same arc or connected (arcs[a,1] == arcs[b,0]); otherwise based on
             network distance vs GPS distance. Use a small floor, never 0.
Work in log space. Split the trip at large time gaps and decode each part separately.
"""
import numpy as np

from pathsnap.geo import project_trip
from pathsnap.metrics import UNLABELLED

LOG_FLOOR = -13.81551  # ln(1e-6)
LOG_SAME_ARC = -0.69315  # ln(0.50)
LOG_CONNECTED = -0.91629  # ln(0.40)


def viterbi(candidates, emission_logp, trip, gap_split_s=60.0):
    """Best candidate arc per GPS point.

    candidates: list of int arrays (per point), emission_logp: list of float arrays
    with the same shapes. Returns an int array of arc ids (-1 where no candidate).
    """
    n_points = len(candidates)
    pred_arc = np.full(n_points, UNLABELLED, dtype=int)
    if n_points == 0:
        return pred_arc

    arcs = trip.arcs
    txy = project_trip(trip)
    seg = txy.seg  # (A, 4): ax, ay, bx, by

    # Identify segment splits based on time gaps
    track = trip.track
    times = track[:, 2]
    finite_mask = np.isfinite(times)
    gaps = np.zeros(n_points, dtype=bool)
    if finite_mask.any():
        dt = np.diff(times)
        gaps[1:] = (dt > gap_split_s) | (dt < 0)

    # Decode segment by segment
    seg_starts = np.flatnonzero(gaps)
    seg_boundaries = np.concatenate([[0], seg_starts, [n_points]])

    for s_idx in range(len(seg_boundaries) - 1):
        start = seg_boundaries[s_idx]
        end = seg_boundaries[s_idx + 1]

        # Points in this segment that have candidate arcs
        valid_indices = [p for p in range(start, end) if len(candidates[p]) > 0]
        if not valid_indices:
            continue

        # Forward pass
        p0 = valid_indices[0]
        n_cands_0 = len(candidates[p0])
        viterbi_scores = [np.asarray(emission_logp[p0], dtype=float)]
        backpointers = [np.full(n_cands_0, -1, dtype=int)]

        for step in range(1, len(valid_indices)):
            prev_p = valid_indices[step - 1]
            curr_p = valid_indices[step]

            prev_cands = candidates[prev_p]
            curr_cands = candidates[curr_p]
            n_curr = len(curr_cands)
            n_prev = len(prev_cands)

            curr_emit = np.asarray(emission_logp[curr_p], dtype=float)
            prev_v = viterbi_scores[-1]

            # Compute transition log-probs matrix: shape (n_prev, n_curr)
            # trans[i, j] is transition from prev_cands[i] to curr_cands[j]
            prev_to = arcs[prev_cands, 1]
            curr_from = arcs[curr_cands, 0]

            # Same arc
            same_mask = prev_cands[:, None] == curr_cands[None, :]
            # Directly connected (prev arc's to_node == curr arc's from_node)
            conn_mask = prev_to[:, None] == curr_from[None, :]

            # Jump distance in metres between prev arc end and curr arc start
            prev_end_x = seg[prev_cands, 2]
            prev_end_y = seg[prev_cands, 3]
            curr_start_x = seg[curr_cands, 0]
            curr_start_y = seg[curr_cands, 1]

            jump_dist = np.hypot(
                prev_end_x[:, None] - curr_start_x[None, :],
                prev_end_y[:, None] - curr_start_y[None, :]
            )

            # Transition probability matrix
            trans_matrix = np.full((n_prev, n_curr), LOG_FLOOR, dtype=float)

            # Distance-decayed transition for nearby arcs
            decay_trans = np.maximum(LOG_FLOOR, -1.6 - (jump_dist / 25.0))
            trans_matrix = np.where(decay_trans > trans_matrix, decay_trans, trans_matrix)
            trans_matrix = np.where(conn_mask, LOG_CONNECTED, trans_matrix)
            trans_matrix = np.where(same_mask, LOG_SAME_ARC, trans_matrix)

            # Combined score: prev_v[:, None] + trans_matrix -> shape (n_prev, n_curr)
            cand_scores = prev_v[:, None] + trans_matrix
            best_prev = np.argmax(cand_scores, axis=0)
            best_scores = cand_scores[best_prev, np.arange(n_curr)] + curr_emit

            viterbi_scores.append(best_scores)
            backpointers.append(best_prev)

        # Backtrack
        last_p = valid_indices[-1]
        best_cand_idx = int(np.argmax(viterbi_scores[-1]))
        pred_arc[last_p] = candidates[last_p][best_cand_idx]

        for step in range(len(valid_indices) - 1, 0, -1):
            curr_p = valid_indices[step]
            prev_p = valid_indices[step - 1]
            best_cand_idx = backpointers[step][best_cand_idx]
            pred_arc[prev_p] = candidates[prev_p][best_cand_idx]

    return pred_arc
