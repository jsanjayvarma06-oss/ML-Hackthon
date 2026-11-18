# Viva preparation

The organisers (Assessment policy section 6) may ask anyone to explain: dataset structure,
preprocessing, feature choices, model comparison and selection, Bayesian optimisation, HMM,
Viterbi, API, UI, and results. Anyone may be asked about any part, so read all sections once;
read your own until you can explain it without the notes. The answers below are short and plain
and describe the code as it is in the repository when this was written. Where a part was not
finished yet it is marked **CHECK**: confirm against the final code before the viva, and never
state a number you have not looked up in `results/`.

## The 60-second story (everyone)
GPS points are noisy. For each point we find nearby road arcs (candidates), describe each
(point, arc) pair with 13 numbers (features), a classifier says how likely each candidate is the
true road, and an HMM with Viterbi picks the best consistent chain of roads so the route is
connected. FastAPI runs this live; a Streamlit UI shows the result on a Folium map.

---

## Sanjay: data and HMM
**Q. What does a trip contain?** Track (lon, lat, time), nodes (junction points), arcs (directed
road pieces between nodes), route (ordered list of arcs actually driven). Ids are local to a trip.
**Q. Why convert to metres?** Degrees are angles; one degree of longitude shrinks with latitude
and trips are worldwide. We project each trip around its own centre (`geo.project_trip`), error is
tiny compared with GPS noise.
**Q. How did you clean the track?** Drop non-finite rows, sort by time, drop repeated/backward
times, drop fixes needing over 70 m/s, split into segments at gaps over 60 s (`clean_track`).
The 5-in-a-row rule stops us deleting a whole trip after one real jump.
**Q. How are candidates found and why K=8?** KD-tree over points every 10 m along arcs, then
exact distance, nearest 8 within 75 m. From `results/pipeline/candidate_miss_rate.csv`: K=1 misses
about 45% (two-way roads give two arcs at equal distance), K=8 about 0.9%, K=16 only 0.4%
better at double the rows. Chosen on train only.
**Q. How were labels made if the route has fewer rows than the track?** Dynamic-programming
alignment (`labels.align_route`): positions never go backwards, minimise total distance, cap the
cost at 100 m so outliers do not dominate. Points over 50 m away are unlabelled and skipped.
**Q. Why 50 m?** Aligned points have median about 4.6 m and p95 about 14 m (`trip_stats.csv`).
**Q. Is the route leaking into features?** No. It only makes `true_arc`/`label` and the API
scoring. `features.py` reads track, nodes, arcs only.
**Q. Why duplicate arcs merged?** Identical from/to arcs would give a coin-flip between equal
candidates (`canonical_arc_ids`).
**Q. What is the HMM here?** Hidden state = which candidate arc the vehicle is on at each point.
Emission = classifier probability. Transition = prefers same or connected arcs; otherwise compares
road distance with GPS distance, with a floor so nothing is impossible. Viterbi = dynamic
programming that finds the single best chain in time proportional to points x candidates squared.
Code is `pathsnap/hmm.py` (`network_distances` runs bounded Dijkstra once, `decode` runs Viterbi per segment); reread it before the viva.
**Q. Why work in log space?** Multiplying hundreds of small probabilities underflows to zero;
adding logs does not.
**Q. Did the HMM help? Numbers?** Train, OOF LightGBM (`hmm_comparison.csv`): nearest arc 0.554
Top-1, classifier only 0.850, distance-only HMM 0.929, classifier + HMM 0.939. Route continuity
0.824, 0.944, 0.981, 0.981.
**Q. So is the classifier useful?** Honestly, it adds about 1 point over a distance-only HMM, and the
tuned emission weight is small (0.1). Most gain is from the transition model. Do not oversell it.
**Q. When did the HMM not help?** On 79 of 80 train trips it beat classifier-only; trip 63 got
worse (0.846 to 0.829). Cause not yet checked (look at the map before claiming one).
**Q. Were HMM parameters tuned fairly?** Grid of beta, floor, weight on train OOF probabilities;
scored on the same trips, so slightly optimistic; top settings within about 0.001, so not sensitive.

## Kushal: models, tuning, UI
**Q. Which five models and why?** Logistic Regression (simple linear baseline), Random Forest,
LightGBM (strong on tabular data), MLP (neural, has epoch history), KNN (instance-based).
All five are in `pathsnap/models.py`. CV Top-1: LightGBM 0.8505, Random Forest 0.8480, MLP 0.8451, KNN 0.8379, LogReg 0.8372.
**Q. What does the classifier predict?** Binary: is this candidate the true arc? We pick the
highest probability candidate per GPS point (Top-1).
**Q. Why GroupKFold by trip?** Points in one trip are strongly related; random splits would put
neighbours of a test point in training and inflate scores.
**Q. Why Macro F1 and log loss?** Most candidates are wrong (class imbalance); macro F1 weighs
both classes equally. Log loss checks probability quality, which the HMM depends on.
**Q. How does Bayesian optimisation work?** Optuna (TPE) builds a model of which settings gave
good scores so far and proposes promising next ones, better than blind grid search. Objective,
ranges, trial count: read `scripts/bayes_opt.py` and `results/models/` trials CSV. **CHECK**.
**Q. Which is better, LogReg or LightGBM?** CV (train): LightGBM Top-1 0.8505 vs 0.8372, F1 0.9105
vs 0.8856, better in all 5 folds, but the gap is small next to fold spread (about 0.055).
**Q. Overfitting?** LightGBM train Top-1 0.879 vs validation 0.851 (mild); LogReg train and
validation nearly equal. Loss curve (fold 0): validation flattens near 0.155 after about 150 trees.
**Q. Did tuning help?** Barely: Optuna (TPE), 40 trials, 3-fold objective; CV Top-1 0.8505 to 0.8510 (+0.0005), well inside the 0.055 fold spread. Default was already near the plateau. Same-data tuning, so mildly optimistic.
**Q. Why is KNN bad?** It memorises: train Top-1 0.991 and train log loss 0.0000 vs validation log loss 0.2536 (worst), and 312.8 ms per 1,000 rows.
**Q. Overfitting evidence?** Gap between train and CV score, loss curves for MLP/boosting only.
**Q. Why no loss curve for Random Forest/KNN?** They have no epoch-by-epoch training; faking one
is forbidden.
**Q. How does the UI work?** Streamlit (`ui/app.py`) calls the API only, shows the map from
`ui/map_view.make_result_map` and the metrics returned. It never reads data files or the route.
**Q. What was student_test used for?** One final evaluation after every choice was frozen.

## Samarth: EDA and API
**Q. What did EDA show?** Only what `results/eda/` shows: sampling rate and gaps, speed spikes,
network sizes, duplicate arcs, self-loops, components. **CHECK**: fill from your own outputs.
**Q. Endpoints?** `GET /health`, `GET /trips`, `POST /predict {trip_id}` (`api/main.py`).
**Q. What happens on /predict?** Load trip, run `pipeline.predict_trip` (clean, candidates,
features, model, HMM), then on the server only read the route to compute accuracy; return GPS,
predicted path segments, metrics, timing. The route is never returned.
**Q. Is anything hardcoded?** No; every request runs the full pipeline live.
**Q. How is the model loaded?** `pipeline.load_model` reads `artifacts/model.joblib` once
(cached), a dict with model, feature columns, name, params.
**Q. Two-laptop setup?** `PATHSNAP_EVAL_DIR` points the API to the trip folder, uvicorn with
`--host 0.0.0.0` so the other laptop can reach it, UI given the API's IP; same hotspot, firewall
port open. Check `README.md`.
**Q. What if the model file is missing?** Should return a clear error (requested). **CHECK**.

## Jagdeesh: report and demo
**Q. Where does every number in the report come from?** A file in `results/`; placeholders in
`final_report.md` name the file. Be ready to open it.
**Q. How did you make sure nothing was fabricated?** Numbers copied only from CSVs, plots made by
our scripts, no curves for models without histories.
**Q. What rules did the team follow?** No evaluator data, student_test used once at the end,
GroupKFold by trip, route never a feature, no hardcoded predictions, seeds fixed (`config.SEED=42`).
**Q. How was AI used?** See `ai_use_disclosure.md`; be able to say it honestly in your own words.
**Q. Walk through the demo.** Start API on Laptop 1, UI on Laptop 2, pick a trip, show map
(GPS vs predicted route), accuracy, timing; show API log lines on Laptop 1.
**Q. Explain one result table.** Pick the 3.1 table: which model won on CV, how it compared with
the nearest-arc baseline and what the HMM added.

## Questions anyone may get
- *Why is Top-1 observation level and not route level?* Both are reported: Top-1 per GPS point,
  route continuity for connectedness.
- *Why can accuracy never reach 100%?* About 0.9% of labelled points have the true arc outside the
  K candidates, and labels are themselves an alignment.
- *What would you improve?* Real answer from the results (e.g. more features, better transition
  model); do not claim improvements you did not test.
