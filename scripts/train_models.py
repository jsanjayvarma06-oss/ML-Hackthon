"""Train and compare the 5+ models. OWNER: Kushal.

Run from the repo root:
    python scripts/train_models.py                          # all models, train CV only
    python scripts/train_models.py --models lightgbm        # a subset
    python scripts/train_models.py --final                  # ONE-TIME student_test run

Default (development) run, train only:
- GroupKFold(5) on train.parquet grouped by trip_id (never a random split).
- Per fold: train/validation macro F1 and log loss (over/underfitting evidence),
  validation Top-1, fit time.
- Out-of-fold probabilities for every train row -> results/models/oof_<model>.parquet
  (contract 5: the HMM is tuned on these, never on student_test).
- Route continuity of the classifier-only OOF prediction, per trip, averaged.
- Writes results/models/model_comparison.csv, cv_folds.csv and plots.

--final (only after every choice is frozen): refit each model on all 80 train trips,
score student_test once -> results/models/final_student_test.csv + confusion matrices.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.base import clone  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, log_loss  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pathsnap import features, io, metrics, models  # noqa: E402
from pathsnap.config import PROCESSED_DIR, RESULTS_DIR, SEED, STUDENT_TEST_DIR, TRAIN_DIR  # noqa: E402

N_SPLITS = 5
THRESHOLD = 0.5  # only for F1 / confusion matrix; Top-1 and the HMM use raw probabilities
OUT = RESULTS_DIR / "models"
KEY_PARAMS = ("C", "class_weight", "n_estimators", "max_depth", "learning_rate", "num_leaves",
              "min_child_samples", "hidden_layer_sizes", "alpha", "n_neighbors", "weights",
              "max_features", "min_samples_leaf", "var_smoothing")


def labelled(df):
    """Training and scoring only use points that have a reference arc."""
    return df[df["true_arc"] != metrics.UNLABELLED].reset_index(drop=True)


def proba(est, X):
    """P(candidate is the true arc) for each row: predict_proba column 1."""
    return est.predict_proba(X)[:, 1]


def key_config(est, name=None):
    """Short hyperparameter string for the comparison table (plus subsample note if any)."""
    sub = models.TRAIN_SUBSAMPLE.get(name)
    base = _key_config(est)
    return base + (f"; trained on fixed subsample of {sub:,} GPS points per fit" if sub else "")


def _key_config(est):
    last = est[-1] if hasattr(est, "steps") else est
    p = last.get_params()
    return "; ".join(f"{k}={p[k]}" for k in KEY_PARAMS if k in p)


def continuity(df, prob, partition_dir, arcs_cache):
    """Mean over trips of route_continuity of the classifier-only (argmax) arc sequence."""
    t = df[["trip_id", "point_idx", "arc_id"]].copy()
    t["prob"] = prob
    best = t.loc[t.groupby(["trip_id", "point_idx"])["prob"].idxmax()]
    vals = []
    for trip_id, g in best.groupby("trip_id"):
        if trip_id not in arcs_cache:
            arcs_cache[trip_id] = io.load_trip(partition_dir, trip_id).arcs
        v = metrics.route_continuity(g.sort_values("point_idx")["arc_id"].to_numpy(),
                                     arcs_cache[trip_id])
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals))


def fit(est, X, y, Xva=None, yva=None):
    """Fit; for LightGBM also record the per-iteration validation log loss."""
    m = clone(est)
    if Xva is not None and m.__class__.__name__ == "LGBMClassifier":
        m.fit(X, y, eval_set=[(X, y), (Xva, yva)], eval_names=["train", "valid"],
              eval_metric="binary_logloss")
    else:
        m.fit(X, y)
    return m


def loss_history(m):
    """Real epoch/iteration histories only (LightGBM eval set, MLP loss_curve_)."""
    if hasattr(m, "evals_result_") and m.evals_result_:
        return {k: v["binary_logloss"] for k, v in m.evals_result_.items()}
    last = m[-1] if hasattr(m, "steps") else m
    if hasattr(last, "loss_curve_"):
        h = {"train": list(last.loss_curve_)}
        if getattr(last, "validation_scores_", None) is not None:
            h["valid_score"] = list(last.validation_scores_)
        return h
    return None


def subsample_points(df, idx, max_points, seed=SEED):
    """Row indices of a fixed random subsample of GPS points (all candidates of a point kept).

    Used only for slow models (models.TRAIN_SUBSAMPLE). idx: row positions of the training
    fold. Returns idx unchanged if the fold has <= max_points points.
    """
    sub = df.loc[idx, ["trip_id", "point_idx"]]
    pid = pd.factorize(pd.MultiIndex.from_frame(sub))[0]
    n_pts = pid.max() + 1
    if n_pts <= max_points:
        return idx
    keep = np.random.default_rng(seed).choice(n_pts, size=max_points, replace=False)
    return idx[np.isin(pid, keep)]


def cross_validate(name, est, df, cols, arcs_cache):
    """GroupKFold(5) by trip_id for one model.

    Returns (per-fold DataFrame with train and validation metrics, out-of-fold probabilities
    for every row, summary dict for model_comparison.csv, fold-0 loss history or None).
    Slow models train on a fixed point subsample (models.TRAIN_SUBSAMPLE); validation is full.
    """
    rows, oof = [], np.zeros(len(df))
    hist = None
    max_pts = models.TRAIN_SUBSAMPLE.get(name)
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=N_SPLITS).split(df, groups=df["trip_id"])):
        if max_pts:  # slow model: fit and report train metrics on the fixed point subsample
            tr = subsample_points(df, tr, max_pts)
        Xtr, ytr = df.loc[tr, cols], df.loc[tr, "label"]
        Xva, yva = df.loc[va, cols], df.loc[va, "label"]
        t0 = time.perf_counter()
        m = fit(est, Xtr, ytr, Xva, yva)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        p_va = proba(m, Xva)
        inf_s = time.perf_counter() - t0
        p_tr = proba(m, Xtr)
        oof[va] = p_va
        if fold == 0:
            hist = loss_history(m)
        rows.append({
            "model": name, "fold": fold, "n_val_trips": df.loc[va, "trip_id"].nunique(),
            "train_f1_macro": f1_score(ytr, p_tr >= THRESHOLD, average="macro"),
            "val_f1_macro": f1_score(yva, p_va >= THRESHOLD, average="macro"),
            "train_log_loss": log_loss(ytr, p_tr, labels=[0, 1]),
            "val_log_loss": log_loss(yva, p_va, labels=[0, 1]),
            "train_top1": metrics.top1_from_table(df.loc[tr], p_tr),
            "val_top1": metrics.top1_from_table(df.loc[va], p_va),
            "fit_seconds": fit_s,
            "infer_ms_per_1000_rows": inf_s / len(va) * 1e6,
        })
        print(f"  fold {fold}: val_top1={rows[-1]['val_top1']:.4f} "
              f"val_f1={rows[-1]['val_f1_macro']:.4f} val_ll={rows[-1]['val_log_loss']:.4f} "
              f"({fit_s:.1f}s)", flush=True)
    cv = pd.DataFrame(rows)
    rep = metrics.classification_report(df["label"], oof, THRESHOLD)
    summary = {
        "model": name,
        "cv_top1_mean": cv.val_top1.mean(), "cv_top1_std": cv.val_top1.std(),
        "cv_f1_macro_mean": cv.val_f1_macro.mean(), "cv_f1_macro_std": cv.val_f1_macro.std(),
        "cv_log_loss_mean": cv.val_log_loss.mean(),
        "cv_train_top1_mean": cv.train_top1.mean(),
        "cv_train_f1_macro_mean": cv.train_f1_macro.mean(),
        "cv_train_log_loss_mean": cv.train_log_loss.mean(),
        "oof_accuracy": rep["accuracy"], "oof_precision_macro": rep["precision_macro"],
        "oof_recall_macro": rep["recall_macro"],
        "oof_route_continuity": continuity(df, oof, TRAIN_DIR, arcs_cache),
        "fit_seconds_mean": cv.fit_seconds.mean(),
        "infer_ms_per_1000_rows": cv.infer_ms_per_1000_rows.mean(),
    }
    return cv, oof, summary, hist


def plot_history(name, hist, out):
    """Save the real per-iteration/epoch loss history (LightGBM eval set, MLP) as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    losses = {k: v for k, v in hist.items() if k != "valid_score"}
    for k, v in losses.items():
        ax.plot(np.arange(1, len(v) + 1), v, label=f"{k} log loss")
    ax.set_xlabel("iteration / epoch")
    ax.set_ylabel("log loss")
    if "valid_score" in hist:
        # MLP early stopping reports accuracy on its 10% hold-out, not a loss: own axis
        ax2 = ax.twinx()
        v = hist["valid_score"]
        ax2.plot(np.arange(1, len(v) + 1), v, color="tab:green", linestyle="--",
                 label="hold-out accuracy (right axis)")
        ax2.set_ylabel("accuracy")
        ax2.legend(loc="center right")
    ax.set_title(f"{name}: training history (CV fold 0)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / f"loss_curve_{name}.png", dpi=150)
    plt.close(fig)


def bar_plot(df, cols, labels, title, fname, out, err=None):
    """Grouped bar chart of one or more metric columns per model; optional error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(df))
    w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (c, lab) in enumerate(zip(cols, labels)):
        ax.bar(x + i * w, df[c], w, label=lab, yerr=None if err is None else df[err[i]])
    ax.set_xticks(x + w * (len(cols) - 1) / 2)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)


def save_confusion(name, cm, out):
    """Save a 2x2 confusion matrix image (student_test, --final only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center")
    ax.set_xticks([0, 1], ["not arc", "arc"])
    ax.set_yticks([0, 1], ["not arc", "arc"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("reference")
    ax.set_title(f"{name}, student_test")
    fig.tight_layout()
    fig.savefig(out / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def upsert(path, new, key="model"):
    """Write `new` rows into a CSV, replacing existing rows with the same key (re-runs are safe)."""
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[~old[key].isin(new[key])]
    out = pd.concat([old, new], ignore_index=True)
    out.to_csv(path, index=False)
    return out


def develop(names, zoo, cols):
    """Development run: GroupKFold CV on train for each model; writes CSVs, OOF files, plots."""
    train = labelled(pd.read_parquet(PROCESSED_DIR / "train.parquet"))
    print(f"train: {train.trip_id.nunique()} trips, {len(train):,} labelled rows")
    plots = OUT / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    arcs_cache = {}
    for name in names:
        family, est = zoo[name]
        print(f"[{name}] GroupKFold({N_SPLITS}) on train ...", flush=True)
        cv, oof, summ, hist = cross_validate(name, est, train, cols, arcs_cache)
        summ = {"model": name, "algorithm_family": family, "key_config": key_config(est, name),
                **{k: v for k, v in summ.items() if k != "model"},
                "n_train_rows": len(train), "seed": SEED}
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()})
        upsert(OUT / "cv_folds.csv", cv)
        comp = upsert(OUT / "model_comparison.csv", pd.DataFrame([summ]))
        o = train[["trip_id", "point_idx", "arc_id"]].copy()
        o["prob"] = oof
        o.to_parquet(OUT / f"oof_{name}.parquet", index=False)
        if hist:
            plot_history(name, hist, plots)
    comp = pd.read_csv(OUT / "model_comparison.csv")
    bar_plot(comp, ["cv_f1_macro_mean"], ["CV macro F1"],
             "Cross-validation macro F1 (GroupKFold by trip, train)", "cv_f1_macro.png", plots,
             err=["cv_f1_macro_std"])
    bar_plot(comp, ["cv_top1_mean"], ["CV Top-1"],
             "Cross-validation Top-1 accuracy (GroupKFold by trip, train)", "cv_top1.png", plots,
             err=["cv_top1_std"])
    bar_plot(comp, ["cv_log_loss_mean"], ["CV log loss"], "CV log loss (lower is better)",
             "cv_log_loss.png", plots)
    print(f"Wrote {OUT / 'model_comparison.csv'}")


def final(names, zoo, cols, extra=None):
    """One-time student_test evaluation, after every choice is frozen."""
    train = labelled(pd.read_parquet(PROCESSED_DIR / "train.parquet"))
    test = labelled(pd.read_parquet(PROCESSED_DIR / "student_test.parquet"))
    plots = OUT / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    items = [(n, *zoo[n]) for n in names] + list(extra or [])
    rows, arcs_cache = [], {}
    for name, family, est in items:
        t0 = time.perf_counter()
        tr_full = train
        if name in models.TRAIN_SUBSAMPLE:
            tr_full = train.loc[subsample_points(train, np.arange(len(train)),
                                                 models.TRAIN_SUBSAMPLE[name])]
        m = fit(est, tr_full[cols], tr_full["label"])
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = proba(m, test[cols])
        inf = (time.perf_counter() - t0) / len(test) * 1e6
        rep = metrics.classification_report(test["label"], p, THRESHOLD)
        rows.append({
            "model": name, "algorithm_family": family,
            "test_top1": metrics.top1_from_table(test, p),
            "test_accuracy": rep["accuracy"], "test_precision_macro": rep["precision_macro"],
            "test_recall_macro": rep["recall_macro"], "test_f1_macro": rep["f1_macro"],
            "test_log_loss": rep["log_loss"],
            "test_route_continuity": continuity(test, p, STUDENT_TEST_DIR, arcs_cache),
            "train_seconds_full": train_s, "infer_ms_per_1000_rows": inf,
            "n_test_rows": len(test),
        })
        print(rows[-1], flush=True)
        save_confusion(name, confusion_matrix(test["label"], p >= THRESHOLD, labels=[0, 1]), plots)
    res = upsert(OUT / "final_student_test.csv", pd.DataFrame(rows))
    bar_plot(res, ["test_top1", "test_f1_macro"], ["Top-1", "Macro F1"],
             "Student-test Top-1 and macro F1 (one final run)", "student_test_top1_f1.png", plots)
    bar_plot(res, ["test_log_loss"], ["Log loss"], "Student-test log loss (lower is better)",
             "student_test_log_loss.png", plots)


def main(argv=None):
    """CLI entry point (see module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="subset of model names (default: all)")
    ap.add_argument("--final", action="store_true",
                    help="one-time student_test evaluation (only after all choices are frozen)")
    args = ap.parse_args(argv)
    cols = list(features.FEATURE_COLUMNS)
    zoo = models.get_models()
    names = args.models or list(zoo)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.final:
        extra = []
        best = OUT / "best_params.json"  # written by bayes_opt.py; scores the tuned model too
        if best.exists():
            import json
            b = json.loads(best.read_text())
            extra.append((f"{b['model']}_tuned", zoo[b["model"]][0], models.build(b["model"], b["params"])))
        final(names, zoo, cols, extra)
    else:
        develop(names, zoo, cols)


if __name__ == "__main__":
    main()
