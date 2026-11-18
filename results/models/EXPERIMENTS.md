# Experiment log: models and tuning (owner: Kushal)

Each entry: command, code commit, date, outputs, headline numbers. Numbers are copied from the output files.
All development runs use train only (GroupKFold(5) by trip_id, SEED=42). student_test is scored once at the end with `--final`.

## M1. Logistic Regression + LightGBM (milestone 1, untuned defaults)
- Command: `python scripts/train_models.py` (models: logistic_regression, lightgbm)
- Commit: `Add logistic regression and LightGBM with GroupKFold training` on top of features v1
- Date: 2026-11-18
- Data: `data/processed/train.parquet`, labelled rows only (1,189,495 rows, 80 trips)
- Outputs: `model_comparison.csv`, `cv_folds.csv`, `oof_<model>.parquet` (out-of-fold probs for HMM tuning; not committed, regenerate with the command), `plots/cv_top1.png`, `plots/cv_f1_macro.png`, `plots/cv_log_loss.png`, `plots/loss_curve_lightgbm.png`
- Headline (CV mean over 5 folds):

| model | CV Top-1 | CV macro F1 | CV log loss | train Top-1 | OOF route continuity | fit s/fold |
|---|---|---|---|---|---|---|
| logistic_regression | 0.8372 ± 0.0569 | 0.8856 | 0.1622 | 0.8376 | 0.9094 | 1.7 |
| lightgbm | 0.8505 ± 0.0551 | 0.9105 | 0.1121 | 0.8788 | 0.9287 | 9.7 |

- Reading: LightGBM beats LogReg on every CV metric. Its train vs validation gap (Top-1 0.879 vs 0.851) is a mild overfit signal; LogReg has almost no gap (underfits: linear boundary). Fold spread (±0.055) is larger than the gap between the models, so per-fold comparison (`cv_folds.csv`) matters.

## M2. Random Forest, MLP, distance-weighted KNN (untuned defaults)
- Command: `python scripts/train_models.py --models random_forest mlp knn_distance_weighted`
- Commit: `Add random forest, MLP, KNN; train all five models`
- Date: 2026-11-18
- Data: same as M1 (1,189,495 labelled train rows, 80 trips, GroupKFold(5) by trip_id, SEED=42)
- Outputs: `model_comparison.csv` and `cv_folds.csv` (rows upserted), `oof_<model>.parquet`, `plots/loss_curve_mlp.png`, refreshed `plots/cv_*.png`
- Training budget: KNN trains on a fixed random subsample of 40,000 GPS points per fold (all candidate rows of a chosen point kept together); MLP on 100,000 points per fold. Validation folds are never subsampled. Sizes were set by compute budget before seeing results. RF uses 150 trees, max_depth=14, min_samples_leaf=20, max_features=sqrt, max_samples=0.5.
- Headline (CV mean over 5 folds):

| model | CV Top-1 | CV macro F1 | CV log loss | train Top-1 | train log loss | OOF route continuity | fit s/fold |
|---|---|---|---|---|---|---|---|
| random_forest | 0.8480 ± 0.0559 | 0.9092 | 0.1157 | 0.8622 | 0.1021 | 0.9209 | 37.1 |
| mlp | 0.8451 ± 0.0563 | 0.9071 | 0.1190 | 0.8563 | 0.1082 | 0.9276 | 26.3 |
| knn_distance_weighted | 0.8379 ± 0.0549 | 0.9021 | 0.2536 | 0.9908 | 0.0000 | 0.9191 | 0.4 |

- Reading: KNN memorises its training subsample (train Top-1 0.991, train log loss 0.0000, train points are their own neighbours) but validates worst on log loss (0.2536): clear overfitting and poorly calibrated probabilities, and slowest at inference (312.8 ms per 1000 rows). RF and MLP sit close to LightGBM (Top-1 within 0.006), all within the ±0.055 fold spread.
- Model to tune: LightGBM (highest CV Top-1 0.8505, also lowest CV log loss 0.1121).

## M3. Optuna (TPE) tuning of LightGBM + tuned vs untuned
- Command: `python scripts/bayes_opt.py --trials 40` (model chosen automatically: best CV Top-1, tie-break log loss, from `model_comparison.csv`)
- Commit: `Add Optuna tuning ...` (see git log), date 2026-11-18, ~20 min on 4 CPU cores
- Search: TPESampler(seed=42), 40 trials, objective = mean validation Top-1 over GroupKFold(3) by trip_id on all 80 train trips. Search space: n_estimators 100-600, learning_rate 0.01-0.2 (log), num_leaves 8-128 (log), min_child_samples 10-400 (log), subsample 0.5-1, colsample_bytree 0.5-1, reg_lambda 0.001-30 (log).
- Outputs: `optuna_trials.csv` (all 40 trials: params, 3-fold Top-1, per-fold Top-1, duration), `best_params.json`, `plots/optuna_history.png`, `plots/optuna_param_importance.png`, `tuned_vs_untuned.csv`, `cv_folds_tuned.csv`, `oof_lightgbm_tuned.parquet`, `plots/loss_curve_lightgbm_tuned.png`
- Result: best trial 33, 3-fold Top-1 0.8496 (trial values: min 0.8336, median 0.8492, max 0.8496). Best params: n_estimators=450, learning_rate=0.0326, num_leaves=50, min_child_samples=121, subsample=0.592, colsample_bytree=0.572, reg_lambda=0.197.
- Tuned vs untuned (same 5 folds as train_models.py):

| variant | CV Top-1 | CV macro F1 | CV log loss | train Top-1 | OOF route continuity |
|---|---|---|---|---|---|
| untuned | 0.8505 ± 0.0551 | 0.9105 | 0.1121 | 0.8788 | 0.9287 |
| tuned | 0.8510 ± 0.0549 | 0.9111 | 0.1116 | 0.8762 | 0.9282 |

- Reading: tuning gained +0.0005 Top-1 and -0.0005 log loss, far smaller than the fold spread (±0.055), so it is not a meaningful improvement; the untuned defaults were already near a plateau (most trials land at 0.849). The tuned model has a slightly smaller train/validation gap (0.8762 vs 0.8510 against 0.8788 vs 0.8505). Caveat: tuning and this comparison use the same 80 train trips, so the tuned figure is mildly optimistic. student_test untouched.

## M4. Export tuned model bundle
- Command: `python scripts/export_model.py` -> `artifacts/model.joblib` = tuned LightGBM (best_params.json) refit on all 80 train trips (1,189,495 labelled rows), 1.1 MB. The reload check (probabilities on 20,000 rows, atol 1e-9) passed. `hmm_params` is None because `results/pipeline/hmm_params.json` did not exist at export time; re-run the script once it does.
- Per-model bundles (`artifacts/models/<name>.joblib`, for the UI picker) were exported by the lead thread with the same script (the Random Forest reload check needs atol, because multi-threaded forests sum trees in a varying order).

## FINAL: one-time student_test evaluation (approved by Wolf, 18 Nov ~13:00 IST)
- Command: `python scripts/train_models.py --final` at commit c804a3b (all choices frozen on train before this run)
- Each model refit on all 80 train trips (KNN/MLP on their fixed subsample), scored once on the 10 student_test trips (125,439 labelled candidate rows, 19,132 points)
- Outputs: `final_student_test.csv`, `plots/confusion_<model>.png`, `plots/student_test_top1_f1.png`, `plots/student_test_log_loss.png`
- Headline Top-1 / macro F1 / log loss: LightGBM 0.8634 / 0.9167 / 0.1020; LightGBM tuned 0.8630 / 0.9164 / 0.1023; RF 0.8596 / 0.9157 / 0.1065; MLP 0.8558 / 0.9129 / 0.1079; LogReg 0.8466 / 0.8935 / 0.1517; KNN 0.8460 / 0.9073 / 0.2284
- Not used for any decision.

## v2 KNN rerun at K=16 (train only)
- Command: `python scripts/train_models.py --models knn_distance_weighted` on the K=16 train table (1,862,764 labelled rows, 80 trips). Upserted into `model_comparison.csv` / `cv_folds.csv`; plots redrawn. All five rows of `model_comparison.csv` are now v2 (K=16).
- KNN CV Top-1 0.8371 ± 0.0544 (v1 K=8: 0.8379), macro F1 0.9042, log loss 0.1762 (still the worst), train Top-1 0.9951 vs validation 0.8371: memorises, heavy overfit. Inference 472 ms per 1,000 rows (slowest).
- Ranking unchanged: LightGBM 0.8520 > RF 0.8489 > MLP 0.8459 > LogReg 0.8395 > KNN 0.8371.

## NB1. Gaussian Naive Bayes (untuned, train only)
- Commands: `python scripts/train_models.py --models naive_bayes`, then `python scripts/hmm_eval.py --oof results/models/oof_naive_bayes.parquet` (its outputs were copied to `results/pipeline/naive_bayes/` so the shared LightGBM `hmm_*` files stay unchanged), then a bundle fit with `export_model.fit_bundle("naive_bayes", ...)` -> `artifacts/models/naive_bayes.joblib` (1.8 KB, reload check passed).
- Date: 2026-11-18. Data: v2 train table (K=16), 1,862,764 labelled rows, 80 trips, GroupKFold(5) by trip_id, SEED=42. student_test not touched and not run.
- What it is, simply: for each class ("this arc is the true one" / "it is not") it learns a bell curve (mean and spread) for every feature, then multiplies the per-feature likelihoods together. Multiplying is only correct if the features are independent given the class (the "naive" assumption). Ours are not: distance to the arc, rank among candidates and distance relative to the best candidate all say nearly the same thing, so each fact is counted several times and the output probabilities become too extreme. The ranking within a point can survive this, the probability values do not.
- Model: StandardScaler + GaussianNB(var_smoothing=1e-9). No hyperparameter search (it has one barely relevant knob), so it is not in the Optuna study.
- Result (CV mean over 5 folds, `model_comparison.csv`):

| model | CV Top-1 | CV macro F1 | CV log loss | train Top-1 | OOF route continuity |
|---|---|---|---|---|---|
| naive_bayes | 0.8024 ± 0.0379 | 0.7644 | 0.8410 | 0.8036 | 0.9132 |
| lightgbm (reference, see `model_comparison.csv`) | 0.8520 ± 0.0580 | 0.9157 | n/a here | n/a | n/a |

- Reading: lowest Top-1 of the six models (-0.050 vs LightGBM, bigger than the other gaps) and by far the worst log loss (0.841 vs about 0.11-0.18 for the rest), which is the over-confidence described above. Train and validation Top-1 are almost equal (0.8036 vs 0.8024), so it underfits (too rigid) rather than overfits. OOF precision (macro) is 0.718 against recall 0.903: it flags too many non-true candidates as positive. It is the fastest to fit (0.8 s per fold) and tiny to store.
- Classifier + HMM (`results/pipeline/naive_bayes/hmm_comparison.csv`, HMM parameters re-fitted on NB's own OOF probabilities, grid in `hmm_grid.csv`, chosen beta=15, floor=50, emission_weight=0.03):

| method | Top-1 (pooled) | route continuity |
|---|---|---|
| classifier only (NB) | 0.8020 | 0.9325 |
| classifier + HMM (NB) | 0.9369 | 0.9799 |
| distance + HMM (no classifier) | 0.9422 | 0.9802 |
| classifier + HMM (tuned LightGBM, reference, E5) | 0.9510 | 0.9806 |

- Reading: the HMM lifts NB by +0.135, but the result is below the distance-only HMM (0.9422): NB's over-confident probabilities are not better evidence than plain distance, and the fit picked the smallest emission weight (0.03) on the grid edge, i.e. it wants to trust NB as little as allowed. NB is a useful low baseline for the report, not a candidate for the final model.
- The bundle `artifacts/models/naive_bayes.joblib` carries these NB-specific hmm_params (the other per-model bundles carry the LightGBM-fitted ones). Smoke test: `pipeline.predict_trip` on train trip 4 returns a prediction for all 1,064 points.
