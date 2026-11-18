"""Bayesian hyperparameter optimisation (Optuna TPE) for the best model. OWNER: Kushal.

Run from the repo root:
    python scripts/bayes_opt.py                         # tunes the best model by CV Top-1
    python scripts/bayes_opt.py --model lightgbm --trials 40

What it does (train only, never student_test):
1. Picks the model to tune: highest CV Top-1 in results/models/model_comparison.csv,
   ties (within 0.002) broken by lower CV log loss. Override with --model.
2. Optuna TPE sampler (seed = config.SEED). TPE = Tree-structured Parzen Estimator: it
   models which parameter regions gave good vs bad scores so far and samples where the
   good/bad density ratio is highest (smarter than random or grid search).
3. Objective = mean validation Top-1 over GroupKFold(3) by trip_id on all 80 train trips.
   3 folds (not 5) keeps one trial to ~half a minute; the final tuned-vs-untuned
   comparison below uses the same 5 folds as every other model.
4. Every trial is appended to results/models/optuna_trials.csv as it finishes.
5. Writes best_params.json, plots/optuna_history.png, plots/optuna_param_importance.png,
   tuned_vs_untuned.csv (5-fold CV, same folds as train_models.py) and
   oof_<model>_tuned.parquet.

Caveat (stated in the report): tuning and the tuned-vs-untuned CV use the same 80 train
trips, so the tuned number is slightly optimistic. student_test stays untouched.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import optuna  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

import train_models as tm  # noqa: E402
from pathsnap import features, metrics, models  # noqa: E402
from pathsnap.config import PROCESSED_DIR, SEED  # noqa: E402

OUT = tm.OUT
TUNE_FOLDS = 3


def pick_model():
    """Best model by CV Top-1 (tie within 0.002 -> lower CV log loss). Returns (name, why)."""
    c = pd.read_csv(OUT / "model_comparison.csv")
    top = c.cv_top1_mean.max()
    close = c[c.cv_top1_mean >= top - 0.002].sort_values("cv_log_loss_mean")
    best = close.iloc[0]
    why = (f"{best.model}: CV Top-1 {best.cv_top1_mean:.4f}, CV log loss "
           f"{best.cv_log_loss_mean:.4f} (best Top-1 {top:.4f}; ties within 0.002 -> log loss)")
    return best.model, why


def suggest(trial, name):
    """Search space per model (kept small and sensible; ranges stated in the report)."""
    if name == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 400, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True))
    if name == "random_forest":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 60, 250, step=30),
            max_depth=trial.suggest_int("max_depth", 6, 24),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            max_features=trial.suggest_float("max_features", 0.2, 0.9),
            max_samples=trial.suggest_float("max_samples", 0.2, 0.7))
    if name == "logistic_regression":
        return dict(C=trial.suggest_float("C", 1e-3, 100.0, log=True))
    if name == "mlp":
        hidden = trial.suggest_categorical("hidden", ["32", "64", "64-32", "128-64", "128-64-32"])
        return dict(
            hidden_layer_sizes=tuple(int(x) for x in hidden.split("-")),
            alpha=trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True))
    if name == "knn_distance_weighted":
        return dict(n_neighbors=trial.suggest_int("n_neighbors", 5, 80))
    raise ValueError(name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model to tune (default: best CV Top-1 in model_comparison.csv)")
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args(argv)
    if args.model:
        name, why = args.model, "chosen with --model"
    else:
        name, why = pick_model()
    print(f"Tuning {name} ({why})", flush=True)

    cols = list(features.FEATURE_COLUMNS)
    train = tm.labelled(pd.read_parquet(PROCESSED_DIR / "train.parquet"))
    folds = list(GroupKFold(n_splits=TUNE_FOLDS).split(train, groups=train["trip_id"]))
    max_pts = models.TRAIN_SUBSAMPLE.get(name)
    trials_csv = OUT / "optuna_trials.csv"
    if trials_csv.exists():
        trials_csv.unlink()  # one study per run; the log is rewritten from scratch

    def objective(trial):
        params = suggest(trial, name)
        est = models.build(name, params)
        scores = []
        for tr, va in folds:
            if max_pts:
                tr = tm.subsample_points(train, tr, max_pts)
            m = tm.fit(est, train.loc[tr, cols], train.loc[tr, "label"])
            scores.append(metrics.top1_from_table(train.loc[va], tm.proba(m, train.loc[va, cols])))
        trial.set_user_attr("fold_top1", [round(s, 5) for s in scores])
        trial.set_user_attr("params_built", json.dumps(params, default=list))
        return float(np.mean(scores))

    def log_trial(study, trial):
        row = {"trial": trial.number, "model": name, "value_cv_top1": trial.value,
               "duration_s": trial.duration.total_seconds(),
               "fold_top1": trial.user_attrs["fold_top1"],
               **{f"param_{k}": v for k, v in trial.params.items()}}
        pd.DataFrame([row]).to_csv(trials_csv, mode="a", header=not trials_csv.exists(), index=False)
        print(f"trial {trial.number}: {trial.value:.4f} ({row['duration_s']:.0f}s) {trial.params}",
              flush=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.trials, callbacks=[log_trial])

    best_params = json.loads(study.best_trial.user_attrs["params_built"])
    (OUT / "best_params.json").write_text(json.dumps({
        "model": name, "params": best_params, "best_cv_top1_3fold": study.best_value,
        "n_trials": args.trials, "seed": SEED, "why_this_model": why,
        "tune_folds": TUNE_FOLDS}, indent=2))
    print("best:", best_params, study.best_value)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)
    vals = [t.value for t in study.trials]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(vals, "o", alpha=0.5, label="trial")
    ax.plot(np.maximum.accumulate(vals), label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel(f"CV Top-1 ({TUNE_FOLDS}-fold)")
    ax.set_title(f"Optuna history: {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "optuna_history.png", dpi=150)
    plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1])
        ax.set_title("Hyperparameter importance (fANOVA-style, Optuna)")
        fig.tight_layout()
        fig.savefig(plots / "optuna_param_importance.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # importance needs >1 distinct values; plot is optional
        print("importance plot skipped:", e)

    # tuned vs untuned, 5-fold CV exactly as train_models.py (same folds, same metrics)
    family = models.get_models()[name][0]
    arcs_cache = {}
    tuned = models.build(name, best_params)
    cv, oof, summ, hist = tm.cross_validate(f"{name}_tuned", tuned, train, cols, arcs_cache)
    base = pd.read_csv(OUT / "model_comparison.csv").set_index("model").loc[name]
    rows = [
        {"variant": "untuned", "model": name, "cv_top1_mean": base.cv_top1_mean,
         "cv_top1_std": base.cv_top1_std, "cv_f1_macro_mean": base.cv_f1_macro_mean,
         "cv_log_loss_mean": base.cv_log_loss_mean, "cv_train_top1_mean": base.cv_train_top1_mean,
         "oof_route_continuity": base.oof_route_continuity, "params": base.key_config},
        {"variant": "tuned", "model": name, "cv_top1_mean": summ["cv_top1_mean"],
         "cv_top1_std": summ["cv_top1_std"], "cv_f1_macro_mean": summ["cv_f1_macro_mean"],
         "cv_log_loss_mean": summ["cv_log_loss_mean"],
         "cv_train_top1_mean": summ["cv_train_top1_mean"],
         "oof_route_continuity": summ["oof_route_continuity"], "params": json.dumps(best_params)}]
    pd.DataFrame(rows).to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    cv.to_csv(OUT / "cv_folds_tuned.csv", index=False)
    o = train[["trip_id", "point_idx", "arc_id"]].copy()
    o["prob"] = oof
    o.to_parquet(OUT / f"oof_{name}_tuned.parquet", index=False)
    if hist:
        tm.plot_history(f"{name}_tuned", hist, plots)
    print(pd.DataFrame(rows).drop(columns="params").to_string(index=False))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.0f}s")
