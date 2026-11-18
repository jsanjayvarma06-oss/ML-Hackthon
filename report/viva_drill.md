# PathSnap viva drill

How to use: cover the answer, say yours out loud, compare. Answers are short on purpose; say them in
your own words. Numbers are real (file named) unless marked **pending final re-run**; if the team re-runs
the final model, those numbers may move, so open the file and quote it, do not recite from memory.
"Train" = 80 trips, cross-validation only. "Student-test" = 10 trips, scored once, after choices were frozen.

---
## Sanjay: data, cleaning, candidates, features, HMM

**1. What is in one trip, and what are we predicting?**
Track (lon, lat, time), nodes, directed arcs, and a reference route (ordered arcs). For every GPS point we predict which arc the vehicle was on, then the chain must be connected. Ids are local to each trip (`pathsnap/io.py`).

**2. Why convert to metres?**
Longitude/latitude are angles; a degree of longitude shrinks with latitude and trips are worldwide. `geo.project_trip` projects each trip around its own centre; error is far below GPS noise.

**3. How do you clean the track?**
`candidates.clean_track`: drop non-finite rows, sort by time, drop repeated/backward times, drop fixes needing over 70 m/s, split at gaps over 60 s. Train: 208,417 raw, 171 spike removals, 208,246 kept, 149 segments (`results/pipeline/trip_stats.csv`).

**4. Why the "5 drops in a row" rule?**
Otherwise one real jump would make every later point look like a spike and delete the rest of the trip. After 5 drops we trust the track again (`MAX_SPIKE_RUN`).

**5. Why split at 60 s gaps?**
Across a gap the vehicle could be anywhere, so linking points would be meaningless. 22 of 80 train trips have such gaps (largest 153,064 s in trip 90).

**6. How are candidates found, and why K=16 now (was K=8)?**
`candidates.generate_candidates`: KD-tree over points every 10 m on arcs, exact distance, nearest K within 75 m. Miss rate (true arc not in candidates): K=1 44.6%, K=3 5.0%, K=8 0.92%, K=16 0.48% (`candidate_miss_rate.csv`). v1 used K=8 as the cost/benefit point; v2 uses K=16 because the miss rate caps accuracy when we aim for 95%. The rows double (about 1.93 million on train).

**7. Why does K=1 miss almost half the time?**
A two-way road is two arcs at the same distance, so the nearest-arc rule is a coin flip on direction. That is why heading is a feature.

**8. How did you make labels from the route?**
`labels.align_route`: dynamic programming; each point gets a route position, positions never go backwards, total distance minimised (cost capped at 100 m). Points over 50 m from their aligned arc stay unlabelled. 185,107 of 208,246 kept points are labelled.

**9. Why a 50 m label cut-off?**
Aligned points have median about 4.6 m and 95th percentile about 14 m distance (`trip_stats.csv`), so 50 m keeps real GPS noise but drops points the route does not cover.

**10. Is the route used as a feature anywhere?**
No. It only creates `true_arc`/`label` (`labels.py`) and the API's server-side scoring. `features.py` reads track, nodes and arcs only.

**11. Name the features and justify two.**
13 features (`features.FEATURE_COLUMNS`). `heading_diff_deg` separates the two directions of a road; `prev_dist_m`/`next_dist_m` check that neighbouring points also sit near this arc, which wrong parallel roads fail. We did not run an ablation, so we cannot claim each feature is needed.

**12. Explain the HMM in plain words.**
Hidden state = which candidate arc the vehicle is on. Emission = classifier probability. Transition = prefers moving along the road about as far as the GPS moved, `exp(-|road dist - GPS dist| / beta)`; unreachable gets a fixed floor, never zero (`hmm.py`). Viterbi finds the single best chain by dynamic programming, in log space.

**13. Why log space?**
Multiplying hundreds of small probabilities underflows to zero; adding logs does not.

**14. How are gaps and disconnected steps handled?**
Segments are decoded separately; an unreachable transition costs the floor, so one bad point cannot kill the path. Points without candidates get -1.

**15. What did the HMM add? (hard)**
v2 train OOF: classifier only 0.850 Top-1, classifier + HMM 0.951, distance-only HMM 0.942 (`hmm_comparison.csv`); v1 was 0.851 / 0.939 / 0.929. Student-test (v1 only, v2 pending): 0.863, 0.921, 0.937 (`hmm_comparison_student_test.csv`). Most of the gain is the HMM, not the classifier. Also be ready for: HMM now bridges time gaps (`hmm.bridge_segments`, up to 5,000 m) and adds a U-turn penalty of 100 for switching to the reverse twin arc.

---
## Samarth: API, deployment, EDA

**1. What did EDA find?**
Train, 80 trips (`results/eda/trip_inventory.csv`): 22 trips with gaps over 60 s, 10 with speed spikes over 70 m/s, 53 with duplicate arcs, 8 with self-loops, 10 with unconnected reference steps, median 20 connected components per network, 649 to 307,392 arcs.

**2. Any missing values?**
None in the raw files: 0 non-finite rows, 0 bad time steps, 0 invalid route ids (`missing_values.csv`). The problems are messy content, not missing cells.

**3. How did EDA change the pipeline?**
Gaps led to segments, spikes to the speed filter, duplicates to canonical arcs, huge networks to the KD-tree, and tied two-way roads to the heading feature.

**4. Which partition was EDA run on?**
Train only; student_test and evaluator trips were not read (`results/eda/EXPERIMENTS.md`).

**5. List the API endpoints.**
`GET /health`, `GET /trips`, `POST /predict {trip_id}`, plus `GET /models`, `GET /comparison` and an optional `model` in /predict (`api/main.py`).

**6. What happens on one /predict call?**
Load the trip, run `pipeline.predict_trip` (clean, candidates, features, model, HMM), then on the server only, read the route to score it. Return GPS, predicted segments, metrics, timing. The route is never returned.

**7. Is anything hardcoded?**
No. Every request runs the full pipeline live. Evaluator labels are never in the student-side pipeline.

**8. How is the model loaded?**
`pipeline.load_model` reads `artifacts/model.joblib` once and caches it; it holds the estimator, feature column order, name, params and HMM params.

**9. What if the model file is missing?**
The API returns a clear error instead of crashing (added in the API change; confirm by running it).

**10. Two-laptop setup, step by step.**
Laptop 1: `PATHSNAP_EVAL_DIR=<trip folder> uvicorn api.main:app --host 0.0.0.0 --port 8000`. Laptop 2: `streamlit run ui/app.py`, enter Laptop 1's IP. Same hotspot, firewall port open. Exact steps are in `README.md`.

**11. Why `--host 0.0.0.0`?**
The default only listens on the same machine; 0.0.0.0 accepts connections from the other laptop.

**12. What shows in the Laptop 1 command window?**
Request log lines per call (trip id, Top-1, continuity, milliseconds), which we screenshot for report section 7.

**13. How do you know the API output matches the offline pipeline?**
Both call the same `features.trip_table` and `predict_trip`, so training and serving features are computed identically. A direct comparison on one trip is the check to run before the demo. **(Do this dry run.)**

**14. What does the API do if a trip has no candidates or odd data?**
Points without candidates get -1 and are skipped in the drawn path; the trip still returns. Test one odd trip (for example 90 with its 42 h gap) before the demo.

**15. Where does inference time come from?**
`timing_ms` in the response (per step and total). Model-only prediction is about 4 ms per 1,000 rows for LightGBM (`final_student_test.csv`).

---
## Kushal: models, CV, Optuna, UI

**1. Which five models, why?**
Logistic Regression (linear baseline), Random Forest (bagging), LightGBM (boosting), MLP (neural), KNN (instance-based), `pathsnap/models.py`. Different families so the comparison means something.

**2. What does the classifier predict?**
For one (point, candidate arc) row: P(this is the true arc). Top-1 picks the highest-probability candidate per point.

**3. Why GroupKFold by trip? (hard)**
Points in one trip are strongly related; with a random split, neighbours of a validation point sit in training and inflate scores. Whole trips are held out, like the real task of an unseen trip (`train_models.py`, 5 folds, seed 42).

**4. Why Macro F1 and log loss as well as Top-1?**
Most candidates are wrong (class imbalance), macro F1 weighs both classes equally. Log loss measures probability quality, which the HMM uses as emission.

**5. Report the CV comparison.**
v2 CV Top-1 (K=16): LightGBM 0.8520, Random Forest 0.8489, MLP 0.8459, LogReg 0.8395; KNN 0.8371 (all rows are v2 now; v1 table in `v1_k8/`). Fold standard deviation is about 0.055 to 0.058, much larger than the 0.014 spread between models, so differences of thousandths are not significant.

**6. Student-test results? (v1 K=8 run; v2 run pending approval)**
Top-1 / F1 / log loss: LightGBM 0.8634 / 0.9167 / 0.1020; RF 0.8596 / 0.9157 / 0.1065; MLP 0.8558 / 0.9129 / 0.1079; LogReg 0.8466 / 0.8935 / 0.1517; KNN 0.8460 / 0.9073 / 0.2284 (`final_student_test.csv`). Quote the file if a final re-run changes them (**pending final re-run**).

**7. Why did KNN overfit? (hard)**
In training each point is its own nearest neighbour: train Top-1 0.995 and train log loss 0.0000 versus validation Top-1 0.8371 and log loss 0.1762, the worst (v2). It memorises, so its probabilities are overconfident, which is bad for an HMM emission. It is also the slowest at prediction (472 ms per 1,000 rows).

**8. Over/underfitting evidence for LightGBM and LogReg?**
LightGBM train Top-1 0.879 vs validation 0.851 (mild overfit); LogReg 0.8376 vs 0.8372 (no overfit, but a linear boundary limits it). Evidence is the per-fold train/validation columns in `cv_folds.csv`.

**9. Which models get loss curves, and why only those?**
LightGBM (eval set) and MLP (epoch history). Random Forest, KNN and LogReg have no epoch-by-epoch training, and fake curves are forbidden.

**10. How does Optuna's Bayesian optimisation work?**
TPE tracks which parameter regions scored well vs badly and proposes settings where good is most likely, so it needs fewer trials than a grid. 40 trials, 3-fold objective, seed 42, all trials in `optuna_trials.csv` (`scripts/bayes_opt.py`).

**11. Why did tuning not help? (hard)**
v1: CV Top-1 0.8505 to 0.8510 (+0.0005), student-test 0.8634 to 0.8630 (-0.0004). v2 (params reused, not re-searched): 0.852 vs 0.852. No measurable difference either time. Most trials landed near 0.849, so the defaults were already near the plateau for these 13 features; gains must come from features or the HMM, not hyperparameters. Tuning also used the same train trips (no nested CV), so even the small CV gain is slightly optimistic.

**12. Then why submit the tuned model?**
That was decided on CV before the student_test run and we did not switch after seeing it, because student_test must never drive a choice. Either model performs the same.

**13. Which model did you tune and how was it chosen?**
Rule written before tuning: highest CV Top-1, ties within 0.002 broken by lower log loss. LightGBM wins both.

**14. How does the UI work?**
Streamlit (`ui/app.py`), talks only to the API, draws GPS and predicted route with `ui/map_view.make_result_map`. It has a model selector and a "Model comparison" page. It never reads data files or the reference route.

**15. Where did student_test come in? (hard)**
Once, with `train_models.py --final`, after every choice was frozen. It was never used for features, K, thresholds, hyperparameters or model choice.

---
## Jagdeesh: report, metrics, limitations

**1. Where does every number in the report come from?**
A file in `results/`; each placeholder or number names its file. Be ready to open it. If a number is not in a file, it must not be in the report.

**2. Why keep the classifier when distance-only HMM won on student_test? (hard)**
Student-test Top-1: distance + HMM 0.9368, classifier + HMM 0.9214. On train CV the classifier + HMM led by about 1 point (0.9388 vs 0.9289). With only 10 test trips the gap may be noise, but we cannot claim the classifier helps once the HMM is applied. We decided before the run and did not change afterwards. The classifier is still far better than nearest arc (0.863 vs 0.535). Cause undiagnosed. (**pending final re-run**: check hmm_comparison_student_test.csv.)

**3. Why not 95% or higher? (hard)**
Four honest limits: (a) 0.9% of labelled points have the true arc outside the 8 candidates; (b) labels are themselves an alignment, and some reference routes are disconnected or do not cover the track; (c) GPS noise near parallel roads and junctions; (d) the HMM tuned on a small grid with 4 parameters. Student-test classifier + HMM was 0.921, distance + HMM 0.937. Quote the pending final number.

**4. What is Top-1 vs route continuity?**
Top-1: share of labelled points whose predicted arc equals the reference arc (`metrics.top1_accuracy`). Continuity: share of consecutive predicted arcs that are the same or connected (`metrics.route_continuity`). A route can be connected and still wrong, so we report both.

**5. Why is student-test higher than CV?**
Classifier Top-1 is 0.863 on student-test vs 0.851 in CV. Only 10 trips and a fold spread of about 0.055, so we do not read anything into it. We did not investigate further.

**6. The demo numbers on trips from `data/raw/train` look great. Are those honest? (hard)**
Those are in-sample: the model was trained on those trips, so numbers there are inflated and must not be presented as generalisation. Honest generalisation numbers are the CV and student-test results. Evaluator trips were never used in any way, so numbers on them are the fair test. Maps for trips 5, 39, 46, 63 in the report are train trips and are labelled in-sample.

**7. How was AI used?**
Claude Code wrote and edited code under each owner's direction and drafted report structure; the team ran every experiment and every number comes from our own result files (`report/ai_use_disclosure.md`). Say it in your own words and be able to defend your part.

**8. How did you prevent leakage?**
Evaluator trips never touched; student_test scored once; GroupKFold by trip; route only for labels and API scoring; seed 42; no hardcoded predictions.

**9. What are the main limitations? (be upfront)**
No feature ablation; tuning on the same trips as comparison (no nested CV); HMM parameters tuned and scored on the same train trips; MLP/KNN trained on subsamples for compute; only 10 student_test trips; HMM classifier benefit not demonstrated on student_test.

**10. What would you improve next?**
Run a feature ablation; nested CV or a held-out tuning set for the HMM; diagnose trips 63 and 33; look at student_test failures only after submission, never to retune.

**11. Walk through the pipeline in 60 seconds.**
Clean the track, find 8 candidate arcs per point, compute 13 features, classifier gives P(correct), HMM + Viterbi picks the best connected chain, FastAPI serves it, Streamlit draws it on a Folium map.

**12. Explain one table or plot.**
Pick the section 3.1 table or `cv_top1.png`: say which model won, by how much, how that compares with fold spread, and how the nearest-arc baseline (about 0.554 train) compares.

**13. How is the Folium map built and what does it show?**
`pathsnap.viz.trip_map` for report maps (GPS, nearby roads, reference route, predicted route, legend, layer control); `ui/map_view.make_result_map` for the UI (GPS and predicted route only).

**14. What would make you distrust a number in the report?**
If it has no file, if it mixes partitions (train vs student_test), or if it was computed after a choice that used the data. Check the file and its partition first.

---
## Everyone: one-line safety answers
- "Did you use the evaluator data?" No, never; not for training, features, tuning or thresholds.
- "Is the route a feature?" No.
- "Can you reproduce it?" Yes: the scripts in the README, seed 42, `results/*/EXPERIMENTS.md` logs each command.
- "I don't know" is better than a guess. Say what the file shows and what you have not checked.

## UPDATE: v2 pipeline (read this first)
The system now has two versions. **v1** = K=8 candidates and the first HMM; its one-time student_test run is done (numbers above). **v2** = K=16 candidates (miss rate 0.48% instead of 0.92%), the HMM bridges time gaps instead of restarting, and switching to the reverse-direction twin arc costs a U-turn penalty of 100 (HMM settings beta 10, floor 50, emission weight 0.1). Train CV Top-1 / route continuity (`results/pipeline/hmm_comparison.csv`): distance only 0.5536 / 0.8243; distance + HMM 0.9422 / 0.9802; classifier only 0.8504 / 0.9444; **classifier + HMM 0.9510 / 0.9806**. Classifier CV Top-1 (v2, `model_comparison.csv`): LightGBM 0.8520, Random Forest 0.8489, MLP 0.8459, LogReg 0.8395 (KNN row still v1 K=8: 0.8379). Tuning again made no difference (0.852 vs 0.852).
- **Caveats you must say:** (1) HMM parameters were chosen on the same out-of-fold points they are scored on; a held-out check gives 0.948 to 0.951, so v2 is at about 95%, not clearly above. (2) The v2 student_test run is **pending the team's approval**; all student_test numbers in this document are v1. (3) The 95% target was set by the team after seeing the v1 student_test result (there is no such threshold in the official PDFs). All v2 choices used train data only, but a v2 student_test run would no longer be a completely untouched check; say so honestly.
- **Hard question: "You changed the system after seeing student_test. Isn't that leakage?"** Answer: we did not use student_test to choose anything. The goal (95%) was set after we saw the v1 number, and every change (K=16, gap bridging, U-turn penalty) was selected on train out-of-fold data only (`EXPERIMENTS.md` E4, E5). But the motivation came after the look, so we report the v1 student_test number as our clean one-time result and treat any v2 student_test number as less clean.
- Pending final re-run: all v2 student_test numbers.
