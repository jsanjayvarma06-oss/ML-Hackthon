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
