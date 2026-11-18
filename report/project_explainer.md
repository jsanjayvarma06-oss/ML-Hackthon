# PathSnap explained in simple language

For all team members. Numbers are real only where a file in `results/` is named. `PENDING` means
the result does not exist yet; never quote a pending number.

## 1. The problem
A phone/vehicle records GPS points every second or so, but GPS is noisy (5 to 20 m off, sometimes
much worse). A road map is available as nodes (junction points) and arcs (one-direction road
pieces; a two-way road is two arcs). **Task:** for every GPS point, pick the arc the vehicle was
really on, then join them into a connected route. A reference route file tells us the truth for
the training trips only.

## 2. The pipeline in one picture
clean the track -> find candidate arcs for each point -> compute features -> classifier gives a
probability per candidate -> HMM + Viterbi picks the best connected chain -> FastAPI serves it ->
Streamlit UI shows it on a map.

## 3. Step by step
**Cleaning** (`clean_track` in `pathsnap/candidates.py`). Convert degrees to metres per trip
(distance on degrees is wrong and trips are worldwide). Drop bad rows, sort by time, drop repeated
or backward times, drop impossible jumps over 70 m/s. Split the trip where the time gap is over 60
s. Train totals (`results/pipeline/trip_stats.csv`): 208,417 raw points, 171 speed spikes removed,
208,246 kept, 149 segments.

**Network fix-up.** Duplicate arcs are merged, self-loops are ignored.

**Candidates.** For each point keep the 8 nearest arcs within 75 m (KD-tree for speed). Why 8:
`results/pipeline/candidate_miss_rate.csv` shows the true arc is missing for 44.6% of points with
K=1 (two-way roads tie), 5.0% with K=3, 0.92% with K=8, 0.48% with K=16. 8 is the sensible
trade-off. Pooled train miss rate 0.0092.

**Labels.** The route has far fewer rows than the track, so we align route and points with dynamic
programming (positions never go backwards, minimise total distance). Points over 50 m from their
aligned arc are unlabelled and ignored. 185,107 of 208,246 kept points are labelled.

**Features** (13 numbers per point-and-candidate, `pathsnap/features.py`): distance, its rank, its
gap to the nearest candidate, heading difference (separates the two directions of a road), a flag
saying heading is reliable, arc length, position along the arc, speed, number of candidates, "reverse
twin is also a candidate", distance of previous and next point to this same arc, and "continues
the previous point's nearest arc". Designed from the geometry; **no ablation has been run**.

**Classifier.** Each (point, candidate) row gets label 1 if it is the true arc. The model outputs
P(correct). Evaluation is 5-fold GroupKFold by trip (whole trips held out, so neighbouring points
cannot leak). Results so far (`results/models/model_comparison.csv`, CV on train, untuned):
Logistic Regression Top-1 0.8372, Macro F1 0.8856; LightGBM Top-1 0.8505, Macro F1 0.9105.
Nearest-arc baseline Top-1 is about 0.554. LightGBM wins in all 5 folds but the gap (0.013) is
small compared with fold-to-fold spread (about 0.055). LightGBM overfits mildly (train Top-1 0.879
vs validation 0.851). Models 2, 4, 5 (Random Forest, MLP, KNN), Bayesian tuning with Optuna, and all
student-test numbers are `PENDING`.

**HMM + Viterbi** (`pathsnap/hmm.py`). Hidden state = which candidate arc the vehicle is on.
Emission = classifier probability (weighted). Transition = prefers moving along the road about as
far as the GPS moved: exp(-|road distance - GPS distance| / beta); unreachable gets a fixed floor
penalty, never zero. Viterbi is dynamic programming that finds the best whole chain in log space.
Train, 80 trips, out-of-fold probabilities (`results/pipeline/hmm_comparison.csv`):

| Method | Top-1 | Route continuity |
|---|---|---|
| Nearest arc | 0.5536 | 0.8243 |
| Distance-only HMM | 0.9289 | 0.9813 |
| Classifier only | 0.8502 | 0.9442 |
| Classifier + HMM | 0.9390 | 0.9809 |

Honest reading: the HMM does most of the work; the classifier adds about 1 point on top. HMM
parameters (beta 10, floor 25, emission weight 0.1) were tuned and scored on the same train trips,
so slightly optimistic.

**API** (`api/main.py`). `/health`, `/trips`, `/predict {trip_id}`. /predict runs the whole
pipeline live, then (server side only) reads the route to compute accuracy. The route is never sent
to the UI. **UI** (`ui/app.py`, Streamlit): talks to the API only and draws GPS plus predicted
route on a Folium map (`ui/map_view.py`).

## 4. Rules we follow (and can say in the viva)
Evaluator trips and student_test never influence any choice; student_test is scored once at the
end. Cross-validation is by trip. The route is never a feature. No hardcoded predictions. Seed 42.
Every report number comes from a file in `results/`.

## 5. Demo plan
Laptop 1: set `PATHSNAP_EVAL_DIR` to the trip folder, run `uvicorn api.main:app --host 0.0.0.0
--port 8000`. Laptop 2: `streamlit run ui/app.py`, enter Laptop 1's IP, pick a trip, run. Show the
map, accuracy, timing, and the request log on Laptop 1. Check README.md for the exact commands and
firewall/hotspot steps. Dry-run once beforehand. `PENDING`: final model bundle in `artifacts/`.
