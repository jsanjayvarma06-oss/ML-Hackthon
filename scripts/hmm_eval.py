"""Compare distance-only, classifier-only and classifier + HMM. OWNER: Sanjay.

Run from the repo root:
    python scripts/hmm_eval.py --oof results/models/oof_lightgbm.parquet
    python scripts/hmm_eval.py --final        # student_test, ONCE, after everything is frozen

Train mode (default): HMM parameters are searched on the 80 train trips with a small grid.
The classifier probabilities come from out-of-fold (GroupKFold by trip) predictions, so
no trip is scored with a model that saw it. Only the 4 HMM numbers (beta, floor, sigma,
emission weight) are fitted on train, nothing else.
  results/pipeline/hmm_grid.csv         every grid point tried (the trial log)
  results/pipeline/hmm_comparison.csv   top-1 and route continuity per method
  results/pipeline/hmm_per_trip.csv     the same per trip, with the chosen parameters
  results/pipeline/hmm_params.json      chosen parameters (read by --final / the pipeline)

--final mode: reads hmm_params.json and artifacts/model.joblib, runs the full live
pipeline on data/raw/student_test and writes results/pipeline/hmm_comparison_student_test.csv.
Never use its numbers to change anything.
"""
import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap import features, hmm, io, labels, metrics, pipeline  # noqa: E402
from pathsnap.config import ARTIFACTS_DIR, RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR  # noqa: E402

OUT = RESULTS_DIR / "pipeline"
BETAS = [1, 2, 5, 10, 20, 50]
FLOORS = [25, 50, 100, 200]
WEIGHTS = [0.05, 0.1, 0.25, 0.5, 1.0]
DIST_WEIGHTS = [0.25, 0.5, 1.0, 2.0, 4.0]


def _score(true, pred, arcs):
    """(correct, labelled, connected pairs, pairs) for pooled metrics."""
    lab = true != metrics.UNLABELLED
    correct, n_lab = int((true[lab] == pred[lab]).sum()), int(lab.sum())
    seq = pred[pred != metrics.UNLABELLED]
    pairs = max(len(seq) - 1, 0)
    cont = metrics.route_continuity(pred, arcs)
    return correct, n_lab, (0 if pairs == 0 else cont * pairs), pairs


def _grid(oof_path):
    """List of (name, emission kind, params dict) to try."""
    g = []
    # sigma is fixed at its default (10 m): scaling the Gaussian is the same as scaling
    # emission_weight, so searching both would only repeat the same trials.
    for beta, floor, w in itertools.product(BETAS, FLOORS, DIST_WEIGHTS):
        g.append(("distance_hmm", "dist", dict(beta=beta, floor=floor, emission_weight=w)))
    if oof_path:
        for beta, floor, w in itertools.product(BETAS, FLOORS, WEIGHTS):
            g.append(("classifier_hmm", "clf", dict(beta=beta, floor=floor, emission_weight=w)))
    return g


def one_trip(args):
    """Score every grid setting on one train trip (network distances computed once)."""
    trip_id, oof_path = args
    trip = io.load_trip(TRAIN_DIR, trip_id)
    route = io.load_route(TRAIN_DIR, trip_id)
    df, cand_arcs, cand_dist, keep_idx, segment_id, txy = features.trip_table(trip)
    true = labels.make_point_labels(trip, route, keep_idx=keep_idx, txy=txy)
    n = len(trip.track)
    net = hmm.network_distances(cand_arcs, trip, txy, keep_idx, segment_id)
    emis = {"dist": hmm.distance_emission(cand_dist)}
    if oof_path:
        oof = pd.read_parquet(oof_path, filters=[("trip_id", "==", trip_id)])
        prob = df[["point_idx", "arc_id"]].merge(oof[["point_idx", "arc_id", "prob"]],
                                                 on=["point_idx", "arc_id"], how="left")["prob"]
        # The OOF file only has labelled rows (models train on labelled rows). Unlabelled points
        # are never scored; they get the same probability for every candidate (no preference),
        # so the transitions alone decide the path through them.
        prob = prob.fillna(0.5)
        emis["clf"] = hmm.classifier_emission(
            pipeline.split_by_point(df, prob.to_numpy(), cand_arcs, keep_idx))
    rows = []
    # argmax baselines (no HMM)
    for name, kind in [("distance_only", "dist"), ("classifier_only", "clf")]:
        if kind in emis:
            pred = hmm.argmax_arcs(cand_arcs, emis[kind], keep_idx, n)
            rows.append((name, None, *_score(true, pred, trip.arcs)))
    for i, (name, kind, p) in enumerate(_grid(oof_path)):
        pred = hmm.decode(cand_arcs, emis[kind], net, keep_idx, segment_id, n, p)
        rows.append((name, i, *_score(true, pred, trip.arcs)))
    return trip_id, rows


def train_mode(oof_path):
    ids = io.list_trips(TRAIN_DIR)
    with ProcessPoolExecutor(min(4, len(__import__("os").sched_getaffinity(0)))) as ex:
        res = list(ex.map(one_trip, [(t, oof_path) for t in ids]))
    grid = _grid(oof_path)
    recs = []
    for trip_id, rows in res:
        for name, gi, c, nl, k, p in rows:
            recs.append(dict(trip_id=trip_id, method=name, grid_idx=gi, correct=c, n_labelled=nl,
                             cont_ok=k, cont_pairs=p))
    d = pd.DataFrame(recs)
    d["grid_idx"] = d["grid_idx"].fillna(-1).astype(int)
    pooled = d.groupby(["method", "grid_idx"], as_index=False)[["correct", "n_labelled", "cont_ok", "cont_pairs"]].sum()
    pooled["top1"] = pooled.correct / pooled.n_labelled
    pooled["route_continuity"] = pooled.cont_ok / pooled.cont_pairs
    # trial log: every grid point with its parameters
    log = pooled[pooled.grid_idx >= 0].copy()
    for key in ["sigma", "beta", "floor", "emission_weight"]:
        log[key] = [grid[i][2].get(key, np.nan) for i in log.grid_idx]
    OUT.mkdir(parents=True, exist_ok=True)
    log[["method", "sigma", "beta", "floor", "emission_weight", "top1", "route_continuity"]].to_csv(
        OUT / "hmm_grid.csv", index=False)

    best = {}
    for name in log.method.unique():
        row = log[log.method == name].sort_values(["top1", "route_continuity"], ascending=False).iloc[0]
        best[name] = {k: float(row[k]) for k in ["sigma", "beta", "floor", "emission_weight"] if not np.isnan(row[k])}
        best[name]["train_top1"] = float(row.top1)
        best[name]["_grid_idx"] = int(row.grid_idx)
    summary = []
    for name in ["distance_only", "distance_hmm", "classifier_only", "classifier_hmm"]:
        if name in best:
            gi = best[name]["_grid_idx"]
        elif (pooled.method == name).any():
            gi = -1
        else:
            continue
        r = pooled[(pooled.method == name) & (pooled.grid_idx == gi)].iloc[0]
        per = d[(d.method == name) & (d.grid_idx == gi)]
        summary.append(dict(method=name, top1_pooled=r.top1, top1_mean_over_trips=float((per.correct / per.n_labelled).mean()),
                            route_continuity=r.route_continuity, n_labelled_points=int(r.n_labelled),
                            n_trips=len(per), oof_file=str(oof_path) if "classifier" in name else ""))
        per.assign(top1=per.correct / per.n_labelled, route_continuity=per.cont_ok / per.cont_pairs)[
            ["trip_id", "method", "top1", "route_continuity", "n_labelled"] if False else
            ["trip_id", "method", "top1", "route_continuity"]].to_csv(
            OUT / f"hmm_per_trip_{name}.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "hmm_comparison.csv", index=False)
    for b in best.values():
        b.pop("_grid_idx")
    best["fitted_on"] = "train, OOF probabilities, GroupKFold by trip" if oof_path else "train, distance only"
    (OUT / "hmm_params.json").write_text(json.dumps(best, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    print(json.dumps(best, indent=2))


def final_mode():
    """Run once on student_test with frozen parameters and the trained bundle."""
    params = json.loads((OUT / "hmm_params.json").read_text())
    bundle = pipeline.load_model(ARTIFACTS_DIR / "model.joblib")
    clf_p = params.get("classifier_hmm") or params.get("distance_hmm")
    dist_p = params.get("distance_hmm")
    rows = []
    for tid in io.list_trips(STUDENT_TEST_DIR):
        trip = io.load_trip(STUDENT_TEST_DIR, tid)
        route = io.load_route(STUDENT_TEST_DIR, tid)
        out_clf = pipeline.predict_trip(trip, {**bundle, "hmm_params": clf_p})
        out_dist = pipeline.predict_trip(trip, {"hmm_params": dist_p})
        true = labels.make_point_labels(trip, route)
        for name, pred in [("distance_only", out_dist["pred_arc_classifier"]), ("distance_hmm", out_dist["pred_arc"]),
                           ("classifier_only", out_clf["pred_arc_classifier"]), ("classifier_hmm", out_clf["pred_arc"])]:
            c, nl, k, p = _score(true, pred, trip.arcs)
            rows.append(dict(trip_id=tid, method=name, correct=c, n_labelled=nl, cont_ok=k, cont_pairs=p))
    d = pd.DataFrame(rows)
    s = d.groupby("method", as_index=False)[["correct", "n_labelled", "cont_ok", "cont_pairs"]].sum()
    s["top1_pooled"] = s.correct / s.n_labelled
    s["route_continuity"] = s.cont_ok / s.cont_pairs
    s["model"] = bundle.get("name", "")
    s[["method", "top1_pooled", "route_continuity", "n_labelled", "model"]].to_csv(
        OUT / "hmm_comparison_student_test.csv", index=False)
    print(s[["method", "top1_pooled", "route_continuity", "n_labelled"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=None, help="OOF probabilities parquet (trip_id, point_idx, arc_id, prob)")
    ap.add_argument("--final", action="store_true", help="student_test, once, frozen parameters")
    a = ap.parse_args()
    if a.final:
        final_mode()
    else:
        train_mode(a.oof)


if __name__ == "__main__":
    main()
