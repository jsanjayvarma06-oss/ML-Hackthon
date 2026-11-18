# PathSnap: map-matching and route reconstruction

PathSnap matches noisy GPS points to a road network and rebuilds the route the vehicle drove.
Pipeline: clean → candidate arcs → features → classifier → HMM + Viterbi → FastAPI → UI.

## Setup (Python 3.11 or 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the data in `data/raw/` as described in [data/README.md](data/README.md).

## Run order

| Step | Command | Output |
|---|---|---|
| 1 | `python scripts/eda.py` | `results/eda/`, `results/maps/` |
| 2 | `python scripts/build_dataset.py` | `data/processed/*.parquet`, `results/pipeline/` |
| 3 | `python scripts/train_models.py` | `results/models/model_comparison.csv` |
| 4 | `python scripts/bayes_opt.py` | `results/models/optuna_trials.csv` |
| 5 | `python scripts/export_model.py` | `artifacts/model.joblib` |
| 6 | `python scripts/hmm_eval.py` | `results/pipeline/hmm_vs_classifier.csv` |

## Deployment (two laptops)

Laptop 1 (model + API):

```bash
export PATHSNAP_EVAL_DIR=data/evaluator
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Laptop 2 (UI):

```bash
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

Enter `http://<laptop-1-ip>:8000` as the API URL in the sidebar.

## Repo layout

```
pathsnap/   shared library: io, geo, candidates, labels, features, models, hmm, pipeline, metrics, viz
scripts/    runnable steps (see run order)
api/        FastAPI backend
ui/         Streamlit evaluator UI
results/    CSVs, plots, maps from real runs
artifacts/  exported model
docs/       contracts between team members
report/     report sections
```

Team ownership and rules: [CLAUDE.md](CLAUDE.md), [WORK_SPLIT.md](WORK_SPLIT.md).
