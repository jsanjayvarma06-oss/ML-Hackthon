# Contracts between owners

These are the hand-off formats between people. Each side codes against them, so we can
work at the same time without touching each other's files. Change a contract only after
telling the other owners, and update this file in the same commit.

## 1. Candidate table (Sanjay produces, Kushal consumes)

Files: `data/processed/train.parquet`, `data/processed/student_test.parquet`.
They are gitignored, so each person makes them by running `python scripts/build_dataset.py`.

One row per (GPS point, candidate arc).

| column | type | meaning |
|---|---|---|
| `trip_id` | str | e.g. `"00000005"`. Use as the group for GroupKFold |
| `point_idx` | int | row index of the GPS point in the original `.track` file (points removed by cleaning have no rows) |
| `cand_rank` | int | 0 = nearest candidate |
| `arc_id` | int | candidate arc id (local to the trip, canonical: duplicates of the same from/to pair map to the first one) |
| `true_arc` | int | reference arc for this point, `-1` = unlabelled |
| `label` | int | 1 if `arc_id == true_arc`, else 0 |
| *features* | float | exactly the names in `pathsnap.features.FEATURE_COLUMNS` |

Rules:
- Train only on rows where `true_arc != -1`.
- `true_arc` and `label` are answers. They must never be used as features.
- Top-1 accuracy: `pathsnap.metrics.top1_from_table(df, prob)`.

## 2. Model bundle (Kushal produces, Sanjay consumes)

File: `artifacts/model.joblib` (committed to git), a dict saved with `joblib.dump`:

```python
{
    "model": fitted_estimator,          # has predict_proba(X)[:, 1]
    "feature_columns": [...],           # order used in training
    "name": "lightgbm_tuned",
    "params": {...},                    # final hyperparameters
    "trained_on": "train (80 trips)",
}
```

Until this file exists, Sanjay tests the HMM with a distance-only emission
(Gaussian on `dist_m`). Keep that as the "no ML" baseline in the report.

## 3. Pipeline (Sanjay produces, Samarth's API consumes)

`pathsnap.pipeline.predict_trip(trip, bundle) -> dict`

| key | type | meaning |
|---|---|---|
| `pred_arc` | int array, length = GPS points | classifier + HMM arc per point, `-1` = none |
| `pred_arc_classifier` | int array | best classifier candidate per point, no HMM |
| `timing_ms` | dict | step name -> ms |

`pathsnap.labels.make_point_labels(trip, route) -> int array` is used by the API for scoring.

## 4. API (Samarth's API produces, Samarth's UI consumes)

- `GET /health` -> `{"status": "ok", "eval_dir": "..."}`
- `GET /trips` -> `{"trips": ["00000015", ...]}`
- `GET /models` -> `{"models": ["lightgbm_tuned", "logistic_regression", ...], "default": "lightgbm_tuned"}`
  (default = `artifacts/model.joblib`; others = `artifacts/models/<name>.joblib`, same bundle format)
- `POST /predict` with `{"trip_id": "00000015"}` or `{"trip_id": "00000015", "model": "<name>"}` ->

```json
{
  "trip_id": "00000015",
  "model": "lightgbm_tuned",
  "n_points": 1045,
  "gps": [[lat, lon], ...],
  "predicted_path": [[[lat, lon], [lat, lon]], ...],
  "metrics": {"top1_accuracy": 0.0, "top1_accuracy_classifier_only": 0.0,
              "route_continuity": 0.0, "n_scored_points": 0},
  "timing_ms": {"total": 0.0}
}
```

The reference route never appears in any response.
