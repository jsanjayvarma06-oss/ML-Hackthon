# PathSnap explained in simple language

For all team members. Numbers are real only where a file in `results/` is named. Two versions exist: **v1** (K=8 candidates, first HMM; its one-time student_test run is done) and **v2** (K=16 candidates, HMM bridges time gaps and penalises U-turns; train CV only). The team decided not to re-run student_test for v2: the 95% target was set after seeing v1, so a second look would not be a clean test.

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

**Candidates.** For each point keep the nearest K arcs within 75 m (KD-tree for speed). Why K matters: `results/pipeline/candidate_miss_rate.csv` shows the true arc is missing for 44.6% of points with K=1 (two-way roads tie), 5.0% with K=3, 0.92% with K=8, 0.48% with K=16. v1 used K=8; v2 uses K=16 because the miss rate caps accuracy when aiming for 95%.

**Labels.** The route has far fewer rows than the track, so we align route and points with dynamic
programming (positions never go backwards, minimise total distance). Points over 50 m from their
aligned arc are unlabelled and ignored. 185,107 of 208,246 kept points are labelled.

**Features** (13 numbers per point-and-candidate, `pathsnap/features.py`): distance, its rank, its
gap to the nearest candidate, heading difference (separates the two directions of a road), a flag
saying heading is reliable, arc length, position along the arc, speed, number of candidates, "reverse
twin is also a candidate", distance of previous and next point to this same arc, and "continues
the previous point's nearest arc". Designed from the geometry; **no ablation has been run**.

**Classifier.** Each (point, candidate) row gets label 1 if it is the true arc. The model outputs P(correct). Evaluation is 5-fold GroupKFold by trip (whole trips held out, so neighbouring points cannot leak). v2 CV Top-1 (`results/models/model_comparison.csv`): LightGBM 0.8520, Random Forest 0.8489, MLP 0.8459, LogReg 0.8395, KNN 0.8371. Nearest-arc baseline about 0.554. LightGBM wins but gaps (about 0.014) are small next to fold-to-fold spread (about 0.055). LightGBM overfits mildly (train Top-1 0.880 vs validation 0.852), KNN heavily (train 0.995 vs 0.837). Optuna tuning (40 trials, run on K=8 data, params reused in v2) made no measurable difference (0.852 vs 0.852). One-time student_test (v1): Top-1 / F1 / log loss LightGBM 0.8634 / 0.9167 / 0.1020 (`results/models/final_student_test.csv`).

**HMM + Viterbi** (`pathsnap/hmm.py`). Hidden state = which candidate arc the vehicle is on. Emission = classifier probability (weighted). Transition = prefers moving along the road about as far as the GPS moved: exp(-|road distance - GPS distance| / beta); unreachable gets a fixed floor penalty, never zero. v2 also bridges time gaps instead of restarting, and charges 100 nats for switching to the reverse twin arc. Viterbi finds the best whole chain in log space. Train CV Top-1 / continuity (`results/pipeline/hmm_comparison.csv`, v2): nearest arc 0.5536 / 0.8243; distance-only HMM 0.9422 / 0.9802; classifier only 0.8504 / 0.9444; classifier + HMM 0.9510 / 0.9806. v1 student_test (`hmm_comparison_student_test.csv`): 0.5351, 0.9368, 0.8630, 0.9214 Top-1. Honest reading: the HMM does most of the work; the classifier adds about 1 point on train, and on v1 student_test the distance-only HMM was 1.5 points better. HMM parameters were chosen on the same train points they are scored on (held-out check 0.948 to 0.951, `hmm_heldout_split.csv`).

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
firewall/hotspot steps. Dry-run once beforehand. The final model bundle is in `artifacts/model.joblib`.
