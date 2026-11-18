"""Metrics used by everyone. OWNER: shared (done; change only after telling the team)."""
import numpy as np

UNLABELLED = -1


def top1_accuracy(true_arc, pred_arc):
    """Share of labelled GPS points whose predicted arc equals the reference arc.

    true_arc, pred_arc: int arrays, one entry per GPS point. Points with
    true_arc == UNLABELLED are skipped.
    """
    true_arc = np.asarray(true_arc)
    pred_arc = np.asarray(pred_arc)
    mask = true_arc != UNLABELLED
    if not mask.any():
        return float("nan")
    return float((true_arc[mask] == pred_arc[mask]).mean())


def route_continuity(pred_arc, arcs):
    """Share of consecutive predicted arcs that are the same arc or connected.

    Connected means the first arc's to_node is the second arc's from_node.
    pred_arc: int array of arc ids per GPS point (UNLABELLED entries are skipped).
    arcs: (A, 2) array from the trip's .arcs file.
    """
    seq = [a for a in np.asarray(pred_arc) if a != UNLABELLED]
    if len(seq) < 2:
        return float("nan")
    ok = sum(1 for a, b in zip(seq, seq[1:]) if a == b or arcs[a, 1] == arcs[b, 0])
    return ok / (len(seq) - 1)


def candidate_miss_rate(true_arc, candidate_arcs):
    """Share of labelled points whose reference arc is not among their candidates.

    candidate_arcs: list of int arrays, one per GPS point.
    """
    hits = [t in set(c) for t, c in zip(true_arc, candidate_arcs) if t != UNLABELLED]
    return float("nan") if not hits else 1 - sum(hits) / len(hits)


def top1_from_table(df, prob):
    """Top-1 accuracy from a candidate table (see docs/CONTRACTS.md).

    For each (trip_id, point_idx) pick the candidate with the highest prob and
    compare its arc_id with true_arc. Unlabelled points are skipped.
    """
    t = df[["trip_id", "point_idx", "arc_id", "true_arc"]].copy()
    t["prob"] = np.asarray(prob)
    t = t[t["true_arc"] != UNLABELLED]
    best = t.loc[t.groupby(["trip_id", "point_idx"])["prob"].idxmax()]
    return float((best["arc_id"] == best["true_arc"]).mean())


def classification_report(y_true, y_prob, threshold=0.5):
    """Candidate-level metrics for the binary (is this the right arc?) task."""
    from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
    }
