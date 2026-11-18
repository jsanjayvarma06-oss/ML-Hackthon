# Experiment log: data pipeline (owner: Sanjay)

Each entry: command, code commit, date, outputs, headline numbers. Numbers are copied from the output files.

## E1. Build candidate + feature tables (features v1)
- Command: `python scripts/build_dataset.py`
- Commit: features v1 (`Add candidate features v1 and dataset builder`), base main @ 01ebede
- Date: 2026-11-18
- Outputs: `data/processed/train.parquet`, `data/processed/student_test.parquet` (gitignored, rebuilt by the command), `results/pipeline/dataset_summary.csv`
- Settings: K=8 candidates within 75 m, gap split 60 s, label cut-off 50 m, 13 features (`pathsnap.features.FEATURE_COLUMNS`)
- Headline: train 80 trips, 208,417 GPS points, 208,246 kept after cleaning, 194,674 with at least one candidate (1,241,235 rows), 185,107 labelled points, pooled candidate miss rate 0.0092. student_test (descriptive stats only, not used for any choice): 10 trips, 125,554 rows, 19,132 labelled points, miss rate 0.0025.


## E2: HMM parameter search on train (OOF LightGBM probabilities)
- Command: `python scripts/hmm_eval.py --oof results/models/oof_lightgbm.parquet`
  (the OOF file used was `oof_lightgbm.parquet` from the models thread, 5-fold GroupKFold by trip, untuned LightGBM)
- Code: features v1 patch + `pathsnap/hmm.py`, `pathsnap/pipeline.py`, `scripts/hmm_eval.py` (this change)
- Data: 80 train trips, 185,107 labelled points. student_test not used.
- Outputs: `results/pipeline/hmm_grid.csv` (every grid trial), `hmm_comparison.csv`,
  `hmm_per_trip_<method>.csv`, `hmm_params.json`
- Grid: beta {1,2,5,10,20,50} m, floor {25,50,100,200} nats, emission weight
  {0.05..1} (classifier) or {0.25..4} (distance). An earlier narrower grid hit its edge, so it was widened.
- Headline (pooled over labelled points, top-1 / route continuity):
  distance-only 0.5536 / 0.8243; distance + HMM 0.9289 / 0.9813;
  classifier-only 0.8502 / 0.9442; classifier + HMM 0.9390 / 0.9809.
- Chosen classifier_hmm parameters: beta 10, floor 25, weight 0.1 (many other settings
  within 0.001 top-1, see hmm_grid.csv).
- Caveats: 4 HMM numbers were picked on the same train trips they are scored on (small
  optimism). Unlabelled points have no OOF probability and get a uniform probability.
  The OOF file has no rows for unlabelled points, so they are context only, never scored.

## Run 2: full pipeline smoke test
- `predict_trip` with distance-only emission and with a LightGBM bundle fitted on all train
  (temporary, in-sample, only to check that nothing crashes) ran on all 80 train trips without error.
  Slowest trips ~2.2 s end to end. No accuracy was recorded from this run.
