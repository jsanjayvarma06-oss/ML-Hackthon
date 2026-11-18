# Work split: Sanjay, Kushal, Samarth

Each person owns different files (see the ownership table in `CLAUDE.md`), so nobody edits
the same file. Hand-offs happen only through the formats in `docs/CONTRACTS.md`.
Jagdeesh: viva prep, report proofreading, testing the two-laptop demo with Samarth.

Order of dependency: Sanjay's candidate table → Kushal's models → Sanjay's HMM pipeline → Samarth's API/UI.
Nobody waits: each person has independent work to do first.

---

## Sanjay: data pipeline + HMM  (the critical path)

**Before Round 1 (11:30)**
1. `geo.py`: lon/lat → local metres, point-to-segment distance, bearing.
2. `candidates.py`: `clean_track` (sort, dedupe, speed spikes, gap splits) and `generate_candidates` with a spatial index.
3. `labels.py`: route → per-point labels by ordered alignment.
4. Number for Round 1: candidate miss rate on train and student_test.

**After Round 1**

5. `features.py` + `scripts/build_dataset.py`: write both parquet files. **Push as soon as it runs, and tell Kushal.**
6. `hmm.py`: Viterbi with a distance-only emission first (works before any model exists).
7. `pipeline.py`: `predict_trip`. Switch to the real model once `artifacts/model.joblib` is pushed.
8. `scripts/hmm_eval.py`: classifier-only vs HMM table on student_test, plus 2–3 maps.
9. Report: `report/hmm_and_data_pipeline.md` (preprocessing, candidates, labels and their limits, features, HMM answers 1–6).

## Kushal: models + Bayesian optimisation

**Before Round 1 (11:30)**
1. `models.py`: the 5 models (Logistic Regression, Random Forest, LightGBM, MLP, KNN), with scalers in Pipelines and `SEED` everywhere.
2. `scripts/train_models.py`: GroupKFold training, metrics, CSV writer, plots. Test it on a tiny fake table with the contract columns while Sanjay's real table is not ready (don't commit the fake data).

**After Round 1**

3. Pull, run `python scripts/build_dataset.py`, then train all 5 on real data → `results/models/model_comparison.csv` + plots (CV F1, student-test Top-1/F1, log loss, time; MLP loss curve only).
4. Pick the model with numbers. `scripts/bayes_opt.py`: Optuna, 30–50 trials, all trials to CSV, tuned vs untuned table.
5. `scripts/export_model.py`: write `artifacts/model.joblib`. **Push and tell Sanjay.**
6. Optional extra: LightGBM ranker (`lambdarank`) as a 6th experiment.
7. Report: `report/models_and_tuning.md` (comparison table, per-model questions A–C, Bayesian section 1–6).

## Samarth: EDA + maps + API + UI + final report

**Before Round 1 (11:30)**
1. `scripts/eda.py` (or `notebooks/01_eda.ipynb`): trip inventory table, missing values and gaps, sampling intervals, network size and components, coordinate plots.
2. `viz.py`: first Folium map of a trip (GPS, nearby network, reference route), with legend and LayerControl.
3. Check that the API starts: `PATHSNAP_EVAL_DIR=data/raw/student_test uvicorn api.main:app --host 0.0.0.0 --port 8000`, then `/trips` from the UI on another laptop over a phone hotspot.

**After Round 1**

4. `ui/app.py`: trip dropdown, metrics, Folium map of GPS + predicted path (`streamlit-folium`).
5. Polish `api/main.py`: logs, error messages, timing. Test all 10 student_test trips end to end.
6. `README.md`: install, run and two-laptop deployment steps. Screenshots of the UI and API logs.
7. Report: `report/eda_maps_api_ui.md`, then combine all three sections into the final PDF, plus an AI-use disclosure.

---

## Round 1 (11:30) checklist
- [ ] Repo has commits from all three accounts
- [ ] EDA table + one Folium map (Samarth)
- [ ] Candidate miss rate (Sanjay)
- [ ] Model code runs on a sample (Kushal)
- [ ] API `/trips` reachable from a second laptop (Samarth)

## Final (16:30) checklist
- [ ] 5 models compared, Bayesian trials saved, model exported
- [ ] Classifier-only vs HMM table
- [ ] API + UI work for all 10 trips on two laptops, API logs visible in the terminal
- [ ] Repo **public**, README complete, PDF report submitted
