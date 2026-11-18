# Models and tuning (owner: Kushal)

Every number below is copied from a file in `results/models/`; the command that produced each file is in `results/models/EXPERIMENTS.md`. All numbers are train-only: GroupKFold(5) by `trip_id` (80 trips, 1,189,495 labelled candidate rows, `config.SEED` = 42). `student_test` is scored once at the end (`train_models.py --final`) and was never used for any choice.

## 1. The task each model solves
For one (GPS point, candidate road arc) row of 13 features (distance, heading difference, arc length, speed, rank among candidates, ...), output the probability that this candidate is the arc the car was really on. For each point, the candidate with the highest probability is the Top-1 prediction. The HMM then uses these probabilities as emission scores.

## 2. Comparison of the five models (untuned defaults)
Source: `model_comparison.csv`, `cv_folds.csv`; plots `plots/cv_top1.png`, `cv_f1_macro.png`, `cv_log_loss.png`.

| model | family | CV Top-1 | CV macro F1 | CV log loss | train Top-1 | train log loss | route continuity (OOF) | fit s/fold | infer ms / 1000 rows |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | Linear | 0.8372 | 0.8856 | 0.1622 | 0.8376 | 0.1596 | 0.9094 | 1.7 | 0.15 |
| random_forest | Ensemble (bagging) | 0.8480 | 0.9092 | 0.1157 | 0.8622 | 0.1021 | 0.9209 | 37.1 | 2.09 |
| lightgbm | Boosting | **0.8505** | 0.9105 | **0.1121** | 0.8788 | 0.0913 | 0.9287 | 9.7 | 3.77 |
| mlp | Neural network | 0.8451 | 0.9071 | 0.1190 | 0.8563 | 0.1082 | 0.9276 | 26.3 | 0.32 |
| knn_distance_weighted | Instance-based | 0.8379 | 0.9021 | 0.2536 | 0.9908 | 0.0000 | 0.9191 | 0.4 | 312.8 |

Important honesty note: the fold-to-fold standard deviation of Top-1 is about 0.055 for every model (some validation folds contain hard trips), which is much larger than the 0.013 gap between best and worst model. The models are therefore ranked on the same folds, but differences of a few thousandths are not significant. The ranking is consistent: LightGBM leads on Top-1 and log loss.

KNN and MLP are trained on fixed point-level subsamples (KNN 40,000 GPS points, MLP 100,000 points per fold, all candidate rows of a chosen point kept together) for compute reasons; validation folds are never subsampled.

## 3. Per-model notes (Deliverables questions A-C)
For each model: A = why this model, B = over/underfitting from the train vs validation gap in `cv_folds.csv`, C = what it tells us.

- **Logistic regression.** A: simple, fast, calibrated baseline. B: train Top-1 0.8376 vs validation 0.8372, so no overfitting but it underfits (a straight line cannot express "close AND same direction"). C: the floor every model must beat.
- **Random forest.** A: bagging of decorrelated trees, robust to scaling. B: train 0.8622 vs validation 0.8480; mild overfit, limited by depth 14 and min_samples_leaf 20. C: close to LightGBM but 4x slower to fit.
- **LightGBM.** A: boosting handles non-linear interactions and records a real validation loss per iteration (`plots/loss_curve_lightgbm.png`). B: train 0.8788 vs validation 0.8505; the largest gap among the tree models but validation log loss is still the lowest. C: best model, chosen for tuning.
- **MLP.** A: a neural-network family member (64-32 hidden units, early stopping). Real epoch history in `plots/loss_curve_mlp.png`. B: train 0.8563 vs validation 0.8451; small gap. C: competitive and the fastest tree-free predictor, but not better than boosting on tabular features.
- **KNN (distance weighted).** A: instance-based comparison. B: heavily overfit: train Top-1 0.991 and train log loss 0.0000 (each training point is its own nearest neighbour) against validation log loss 0.2536, the worst. C: poorly calibrated probabilities, so a bad HMM emission, and 312.8 ms per 1000 rows at inference.

## 4. Why LightGBM was tuned
Rule (written before tuning, implemented in `bayes_opt.py`): highest CV Top-1, ties within 0.002 broken by lower CV log loss. LightGBM wins on both (0.8505, 0.1121); the runner-up RF is 0.0025 behind on Top-1, outside the tie band.

## 5. Bayesian optimisation (Optuna)
Source: `optuna_trials.csv`, `best_params.json`, `tuned_vs_untuned.csv`, `plots/optuna_history.png`, `plots/optuna_param_importance.png`.

1. **Method:** Optuna TPE (Tree-structured Parzen Estimator), seed 42. It models which parameter regions scored well versus badly and proposes points where good is most likely relative to bad, so it needs far fewer trials than grid search.
2. **Search space:** n_estimators 100-600, learning_rate 0.01-0.2 (log), num_leaves 8-128 (log), min_child_samples 10-400 (log), subsample 0.5-1, colsample_bytree 0.5-1, reg_lambda 0.001-30 (log).
3. **Objective:** mean validation Top-1 over GroupKFold(3) by trip on all 80 train trips (3 folds instead of 5 so a trial costs about 30 s).
4. **Trials:** 40, all logged with params, per-fold score and duration (about 18 minutes of trial time). Trial values: min 0.8336, median 0.8492, max 0.8496. Best is trial 33.
5. **Best parameters:** n_estimators 450, learning_rate 0.0326, num_leaves 50, min_child_samples 121, subsample 0.592, colsample_bytree 0.572, reg_lambda 0.197.
6. **Tuned vs untuned** (same 5 folds): Top-1 0.8505 -> 0.8510, macro F1 0.9105 -> 0.9111, log loss 0.1121 -> 0.1116, train Top-1 0.8788 -> 0.8762, route continuity 0.9287 -> 0.9282.

**Interpretation, stated plainly:** tuning improved Top-1 by only +0.0005, far inside the fold spread (0.055). We conclude the default LightGBM was already near the plateau for these 13 features; most trials landed at about 0.849. The benefit is a slightly smaller train/validation gap, not accuracy. Larger gains must come from features or the HMM, not from hyperparameters. Caveat: tuning and the comparison use the same 80 train trips, so the tuned figure is mildly optimistic; no nested CV was run.

## 6. Exported model
`python scripts/export_model.py` refits the tuned LightGBM on all 80 train trips and writes `artifacts/model.joblib` (1.1 MB; fields: model, feature_columns, name, params, trained_on, hmm_params), after checking that reloading gives the same probabilities. `hmm_params` is None until Sanjay's `results/pipeline/hmm_params.json` exists; re-run the export then.

## 7. Student-test (one-time)
Not yet run. `python scripts/train_models.py --final` scores all five models plus the tuned model once, after every choice is frozen. Results will go to `final_student_test.csv`.
