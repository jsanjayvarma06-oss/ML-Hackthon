# PathSnap: map-matching hackathon (UE24CS352A, 18 Nov 2026)

Team: Sanjay, Kushal, Samarth, Jagdeesh. Sanjay, Kushal and Samarth each work through
their own Claude Code on their own laptop and push from their own GitHub account.

## Task
For every noisy GPS point, pick the road arc the vehicle was on. Then join the arcs into a
connected route. Pipeline: clean -> candidates (k nearest arcs) -> features -> classifier
(probability per candidate) -> HMM + Viterbi -> FastAPI (Laptop 1) -> UI (Laptop 2).
The rubric needs: 5+ models compared, Bayesian optimisation with the trial log kept,
classifier-only vs HMM comparison, Folium maps, and a PDF report.

## Ownership: only edit files you own
| Owner | Files |
|---|---|
| Sanjay | `pathsnap/geo.py`, `candidates.py`, `labels.py`, `features.py`, `hmm.py`, `pipeline.py`, `viz.py`, `ui/map_view.py`, `scripts/build_dataset.py`, `scripts/hmm_eval.py`, `scripts/make_report_maps.py`, `report/hmm_and_data_pipeline.md` |
| Kushal | `pathsnap/models.py`, `scripts/train_models.py`, `scripts/bayes_opt.py`, `scripts/export_model.py`, `artifacts/`, `report/models_and_tuning.md` |
| Samarth | `scripts/eda.py`, `notebooks/`, `api/`, `ui/app.py`, `ui/requirements.txt`, `README.md`, `report/eda_maps_api_ui.md` |
| Shared (ask first) | `pathsnap/config.py`, `pathsnap/io.py`, `pathsnap/metrics.py`, `docs/CONTRACTS.md`, `requirements.txt`, `CLAUDE.md` |

Each person writes outputs only into their own subfolders: Sanjay `results/pipeline/` and
`results/maps/trip_*.html`, Kushal `results/models/`, Samarth `results/eda/` and
`results/maps/eda_*.html`.

Maps: `ui/map_view.make_result_map(result)` draws the API result in the UI (GPS + predicted
route only, never the reference route). `pathsnap.viz.trip_map(...)` draws report maps
(GPS, nearby roads, reference route, predicted route). Use these; don't write new map code.
If a task needs a change in someone else's file, do not edit it. Write the request in
`docs/REQUESTS.md` and tell that person. Hand-off formats live in `docs/CONTRACTS.md`.

## Git workflow
- Work on `main`. Before every push: `git pull --rebase`. Commit small and often.
- Never commit `data/` contents, secrets, or other people's files.

## Hard rules (from the hackathon PDFs)
- Evaluator trips (`data/evaluator/`, ids 15 17 23 30 49 59 67 73 87 94) are never used
  for training, feature selection, tuning or thresholds.
- Route files feed only `labels.py` and the API's scoring. Never use them as features,
  never return them to the UI.
- No hardcoded predictions or trip-specific shortcuts. The API runs the full pipeline live.
- Cross-validation is GroupKFold by `trip_id`. Never use random splits.
- Tuning must be Bayesian (Optuna), and every trial is saved to CSV.
- Loss curves only for models with real epoch histories (MLP, boosting with an eval set).
- Every number in the report must come from a file in `results/`. Use `config.SEED`.
- Do not copy code from public map-matching repos or other teams.

## Data facts (checked on all 90 trips)
- Columns are lon, lat. Folium needs [lat, lon]. Trips are worldwide, so convert to
  metres per trip and never use Euclidean distance on degrees.
- Node and arc ids are local to each trip. Arcs are directed, and most roads have a reverse twin.
- The route is an ordered arc list, not one row per GPS point. Labels come from alignment.
- Sampling is mostly 1 Hz, but there are fractional timestamps (trips 2, 71, 97), big
  gaps (trip 90: about 42 h, trip 78: 52,810 s), and speed spikes over 70 m/s (13, 32, 79).
- Duplicate arcs, self-loops and many disconnected components exist. Some reference routes
  have unconnected steps (2, 19, 33, 41, 42, 69, 75, 80, 86, 89, test 76).
- Network size ranges from 649 to 307,392 arcs, so use a spatial index.

## Commands (run from the repo root, Python 3.11/3.12 venv)
```
pip install -r requirements.txt
python scripts/eda.py
python scripts/build_dataset.py
python scripts/train_models.py
python scripts/bayes_opt.py
python scripts/export_model.py
python scripts/hmm_eval.py
PATHSNAP_EVAL_DIR=data/raw/student_test uvicorn api.main:app --host 0.0.0.0 --port 8000
streamlit run ui/app.py
```
