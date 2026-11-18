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
