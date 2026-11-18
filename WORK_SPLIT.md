# Work split: 3 parallel groups (from 11:45, 18 Nov)

Three groups, each owning different files (ownership table in `CLAUDE.md`), so nobody
edits the same file. Everyone can start right now. Hand-offs happen only through the
formats in `docs/CONTRACTS.md`, merged to `main` through pull requests.
Jagdeesh helps Group C with EDA plots, screenshots, the report and viva prep.

**Already done on main:** loader, metrics, geo, track cleaning, candidates, route labels,
candidate stats (`results/pipeline/`), UI map + report maps, Streamlit UI (`ui/app.py`),
API skeleton.

## Hand-off timeline (everyone works toward these)

| Time | From → To | What lands on `main` |
|---|---|---|
| 12:15 | A → B | `features.py` v1 + `build_dataset.py`. B runs it to get the parquet files |
| 12:40 | A → C | `hmm.py` + `pipeline.predict_trip` with distance-only emission. The API works end to end |
| 14:15 | B → A | `artifacts/model.joblib` (untuned) + out-of-fold probabilities on train |
| 15:00 | B → A | tuned `artifacts/model.joblib` |
| **15:30** | **FREEZE** | No more choices. Each group runs its one `--final` student_test evaluation and commits the results |
| 15:50 | A, B → C | report sections |
| 16:15 | C | PDF done, repo public, both laptops set up |

Lunch 12:45, leadership address 13:30 (at the venue by 13:15). Push before you leave.

---

## Group A: Sanjay, data → route

Files: `pathsnap/features.py`, `hmm.py`, `pipeline.py`, `viz.py`, `ui/map_view.py`,
`scripts/build_dataset.py`, `scripts/hmm_eval.py`, `scripts/make_report_maps.py`,
`report/hmm_and_data_pipeline.md`

1. **features v1 (by 12:15):** `dist_m`, `dist_rank`, `dist_gap_to_best`, `heading_diff_deg`,
   `arc_len_m`, `proj_t`, `speed_mps`, `n_candidates`, `has_reverse_twin`.
   `build_dataset.py` writes `train.parquet` and `student_test.parquet`.
   Branch `sanjay/features`, PR, merge, tell Kushal.
2. **HMM + pipeline v0 (by 12:40):** Viterbi with distance-only emission, transitions from
   connectivity + network distance, split at gaps. `predict_trip` returns the contract keys.
   Merge, tell Samarth.
3. **After lunch:** plug in Kushal's model. Tune the HMM parameters on his out-of-fold train
   probabilities only. `hmm_eval.py` compares classifier-only vs HMM on train (CV).
4. Optional: features v2 (more features, same table format). Only before 14:15.
5. **15:30:** `python scripts/hmm_eval.py --final` → `results/pipeline/final_hmm_vs_classifier.csv`.
   Rerun `make_report_maps.py` so the maps show the red predicted route.
6. Report section: preprocessing, candidates, labels and their limits, features, HMM questions 1–6.

## Group B: Kushal, models

Files: `pathsnap/models.py`, `scripts/train_models.py`, `scripts/bayes_opt.py`,
`scripts/export_model.py`, `artifacts/`, `ui/app.py`, `ui/requirements.txt`,
`report/models_and_tuning.md`

1. **Now (before features land):** `models.py` with 5 models (Logistic Regression, Random
   Forest, LightGBM, MLP, distance-weighted KNN), scalers in Pipelines, `SEED`.
   `train_models.py` with GroupKFold by `trip_id` on train, using
   `metrics.classification_report` and `metrics.top1_from_table`, plus timings and plots.
   Test on a small fake table with the contract columns; don't commit it.
2. **12:15:** pull main, run `build_dataset.py`, train all 5 for real →
   `results/models/model_comparison.csv` + plots (CV macro F1, CV Top-1, log loss, train and
   inference time; loss curve only for the MLP).
3. **Before 14:15:** pick the model on CV numbers. Write out-of-fold probabilities
   (contract 5) and an untuned `artifacts/model.joblib`. Merge, tell Sanjay.
4. **Before 15:00:** `bayes_opt.py` (Optuna, at least 30 trials, objective = CV score on train),
   `optuna_trials.csv`, tuned vs untuned on CV. Export the tuned model. Merge, tell Sanjay.
5. **15:30:** `python scripts/train_models.py --final` → the student-test columns of the
   comparison table (all 5 models + tuned) in `results/models/final_student_test.csv`.
6. Report section: comparison table, the per-model questions A–C, Bayesian questions 1–6.

## Group C: Samarth (+ Jagdeesh), EDA, API, deployment, report

Files: `scripts/eda.py`, `notebooks/`, `api/`, `README.md`, `report/eda_maps_api_ui.md`,
`report/final/`

1. **Now:** `eda.py` on **train**: inventory table, sampling intervals, gaps, speed spikes,
   network size and components, coordinate plots → `results/eda/`. EDA maps via
   `viz.trip_map(trip, route=route)` → `results/maps/eda_<trip>.html`.
2. **Now:** API: add the model name to `/health` (request in `docs/REQUESTS.md`). Test
   `/trips` and `/predict` with `PATHSNAP_EVAL_DIR=data/raw/train` as soon as Sanjay's
   pipeline v0 lands (about 12:40).
3. **Two laptops:** phone hotspot, firewall prompt for port 8000, UI on laptop 2 reaching the
   API on laptop 1. Write the steps in `README.md`. Screenshots of the UI and the API terminal logs.
4. **Demo-day folder:** know how to point `PATHSNAP_EVAL_DIR` at the evaluator trips and
   check that all of them load.
5. **Report:** collect the three sections into `report/final/`, add the AI-use disclosure,
   build the PDF. Every number must match a file in `results/`. Make the repo public before 16:30.

---

## Final (16:30) checklist
- [ ] 5 models compared on CV, Bayesian trials saved, tuned model exported
- [ ] Classifier-only vs HMM table (CV + one final student_test run)
- [ ] API + UI work on two laptops for every evaluator trip, API logs visible
- [ ] Repo public, README complete, PDF submitted
