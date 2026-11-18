# PathSnap: map-matching and route reconstruction

PathSnap matches noisy GPS points to a road network and rebuilds the route the vehicle drove.
Pipeline: clean → candidate arcs → features → classifier → HMM + Viterbi → FastAPI → UI.

## Setup (Python 3.11 or 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data placement

Unzip `Student folder.zip` and put its folders here (they are gitignored, never commit them):

```
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}     80 trips
data/raw/student_test/<trip_id>/...                              10 trips
data/evaluator/<trip_id>/...                                     given on demo day, Laptop 1 only
```

Details and rules: [data/README.md](data/README.md). `data/processed/` is generated.

## Run order

| Step | Command | Output |
|---|---|---|
| 1 | `python scripts/eda.py` | `results/eda/trip_inventory.csv`, `missing_values.csv`, PNG plots, `results/maps/eda_*.html` |
| 2 | `python scripts/build_dataset.py` | `data/processed/*.parquet`, `results/pipeline/` |
| 3 | `python scripts/train_models.py` | `results/models/model_comparison.csv` |
| 4 | `python scripts/bayes_opt.py` | `results/models/optuna_trials.csv` |
| 5 | `python scripts/export_model.py` | `artifacts/model.joblib` |
| 6 | `python scripts/hmm_eval.py` | `results/pipeline/hmm_comparison.csv`, `hmm_params.json` |
| 7 | `python scripts/make_report_maps.py` | `results/maps/trip_*.html` |

## Deployment (two laptops)

Both laptops must be on the same network (same Wi-Fi, or one phone hotspot). Laptop 1 runs
the model and the API. Laptop 2 runs only the UI and talks to Laptop 1 over HTTP.

**Laptop 1 (model + API)** needs the repo, `artifacts/model.joblib`, and the evaluator
trips in `data/evaluator/`. The API reads the reference route only to score a prediction
and never returns it.

```bash
export PATHSNAP_EVAL_DIR=data/evaluator      # folder with the trips to serve
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` makes the API reachable from other machines (the default, 127.0.0.1, is
local only). Find Laptop 1's IP with `ipconfig` (Windows) or `ip addr` / `ifconfig`
(Linux, Mac). Allow inbound TCP 8000 in the firewall if Laptop 2 cannot connect.

**Laptop 2 (UI)**:

```bash
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

In the sidebar, set the API URL to `http://<laptop-1-ip>:8000`.

Check before the demo:

1. On Laptop 2 open `http://<laptop-1-ip>:8000/health` in a browser. It should show
   `"status": "ok"`, the `eval_dir`, and the `model` name.
2. `/health` shows `"model": null` and `"model_file_found": false` until
   `python scripts/export_model.py` has produced `artifacts/model.joblib`. `/predict` then
   answers 503 with that instruction.
3. `/trips` lists the trip ids found in `PATHSNAP_EVAL_DIR`. An empty or wrong folder
   gives a clear 500 message.

The API terminal prints one log line per request (trip id, accuracy, time taken).

For local development you can serve training trips instead:
`PATHSNAP_EVAL_DIR=data/raw/train uvicorn api.main:app --port 8000`.

## Repo map (file: purpose, owner)

| File | Purpose | Owner |
|---|---|---|
| `pathsnap/io.py` | read trip files (track, nodes, arcs, route) | shared |
| `pathsnap/config.py` | paths, `SEED`, `PATHSNAP_EVAL_DIR` | shared |
| `pathsnap/metrics.py` | Top-1 accuracy, route continuity | shared |
| `pathsnap/geo.py` | metre conversion, spatial helpers | Sanjay |
| `pathsnap/candidates.py` | k nearest arcs per GPS point | Sanjay |
| `pathsnap/labels.py` | align GPS to the reference route for labels | Sanjay |
| `pathsnap/features.py` | candidate features | Sanjay |
| `pathsnap/hmm.py` | HMM + Viterbi route reconstruction | Sanjay |
| `pathsnap/pipeline.py` | `predict_trip`: the full live pipeline the API calls | Sanjay |
| `pathsnap/viz.py` | Folium report maps (`trip_map`) | Sanjay |
| `pathsnap/models.py` | the compared classifiers | Kushal |
| `scripts/build_dataset.py`, `candidate_stats.py`, `hmm_eval.py`, `make_report_maps.py` | data build, stats, HMM comparison, report maps | Sanjay |
| `scripts/train_models.py`, `bayes_opt.py`, `export_model.py` | model comparison, Optuna tuning, export | Kushal |
| `scripts/eda.py` | dataset exploration (train only) | Samarth |
| `api/main.py` | FastAPI backend (Laptop 1) | Samarth |
| `ui/app.py` | Streamlit evaluator UI (Laptop 2) | Kushal |
| `ui/map_view.py` | Folium map of an API result | Sanjay |
| `artifacts/model.joblib` | exported model bundle | Kushal |
| `docs/CONTRACTS.md`, `docs/REQUESTS.md` | hand-off formats, cross-owner requests | shared |
| `results/` | CSVs, plots, maps from real runs (per-owner subfolders) | each owner |
| `report/` | report sections, one file per owner | each owner |

Team ownership and rules: [CLAUDE.md](CLAUDE.md), [WORK_SPLIT.md](WORK_SPLIT.md).

## Where each report number comes from

Every number in the report must come from a file in `results/`. Each owner keeps a log of
runs (command, commit, output files, headline numbers) next to their results:

- EDA: `results/eda/EXPERIMENTS.md`
- Models and tuning: `results/models/EXPERIMENTS.md`
- Data pipeline and HMM: `results/pipeline/EXPERIMENTS.md`

Cross-validation is GroupKFold by `trip_id`; the seed is `config.SEED`. student_test is
scored once at the end behind `--final`; the evaluator trips are never used for any choice.
