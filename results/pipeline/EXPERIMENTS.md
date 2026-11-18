# Experiment log: data pipeline (owner: Sanjay)

Each entry: command, code commit, date, outputs, headline numbers. Numbers are copied from the output files.

## E1. Build candidate + feature tables (features v1)
- Command: `python scripts/build_dataset.py`
- Commit: features v1 (`Add candidate features v1 and dataset builder`), base main @ 01ebede
- Date: 2026-11-18
- Outputs: `data/processed/train.parquet`, `data/processed/student_test.parquet` (gitignored, rebuilt by the command), `results/pipeline/dataset_summary.csv`
- Settings: K=8 candidates within 75 m, gap split 60 s, label cut-off 50 m, 13 features (`pathsnap.features.FEATURE_COLUMNS`)
- Headline: train 80 trips, 1,241,235 rows, 194,674 kept points, 185,107 labelled points, pooled candidate miss rate 0.0092. student_test (descriptive stats only, not used for any choice): 10 trips, 125,554 rows, 19,132 labelled points, miss rate 0.0025.
