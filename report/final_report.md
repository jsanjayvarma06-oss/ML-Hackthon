# PathSnap technical report (team skeleton)

> **Status: partly filled (dataset, EDA, preprocessing, models 1 and 3 CV).** Layout follows `Deliverables.pdf` sections 2.2 to 8. Every number is
> either (a) quoted from a file already in `results/` (the file is named next to it), or
> (b) a placeholder written as `[[TODO: value <- results/<file>]]`. Nothing here has been
> measured unless a file is cited. Replace each placeholder only with the real value from
> that file, then delete the marker. Do not invent or round up numbers.
>
> Partitions used throughout: **train** = 80 trips (all tuning, cross-validation, every design
> choice). **student_test** = 10 trips, looked at only once at the very end after every choice
> was frozen. **evaluator** trips are never used for anything.

---

## 2.2 Dataset understanding and exploratory data analysis

### 2.2.1 What the data is (plain language)
Each trip is four text files. `.track` is the noisy GPS trace (lon, lat, time). `.nodes` and
`.arcs` are the road map around that trip: nodes are junction points, an arc is one directed road
piece from one node to another (a two-way road is two arcs). `.route` is the reference answer,
an ordered list of arcs the vehicle really drove. Ids are local to a trip. The task is to say,
for every GPS point, which arc the vehicle was on, then join those arcs into a connected route.

### 2.2.2 How we preprocessed the data (all of this is `pathsnap/candidates.py` and `geo.py`)
1. **Convert degrees to metres.** Longitude/latitude are angles, so distances computed on them
   are wrong, and the trips are spread over the world. Each trip is projected to local x/y
   metres around its own centre (`geo.project_trip`). Over a trip this is accurate to a fraction
   of a percent, far below GPS noise (5 to 20 m).
2. **Clean the GPS track** (`candidates.clean_track`), in this order:
   - drop rows with missing/non-finite values;
   - sort by time and drop repeated or backward timestamps;
   - drop a fix that would need more than **70 m/s (252 km/h)** from the last kept fix (a GPS
     spike). If 5 fixes in a row are dropped, we trust the track again so a real jump does not
     delete the rest of the trip.
   Counts on train (from `results/pipeline/trip_stats.csv`): 208,417 raw points; 0 non-finite;
   0 bad timestamps; 171 speed spikes removed; 208,246 kept.
3. **Split at time gaps.** A gap longer than 60 s starts a new *segment*; the HMM later never
   links points across a gap. 22 of the 80 train trips have more than one segment (149
   segments in total, same file).
4. **Network clean-up.** Exact duplicate arcs (same from/to) are merged to one canonical arc, and
   self-loops are never candidates (`candidates.canonical_arc_ids`, `usable_arcs`). Without this
   two identical roads would be a coin flip.
5. **Candidate generation** (`candidates.generate_candidates`). For each kept GPS point we keep
   the **K nearest arcs within 75 m**. A KD-tree over points sampled every 10 m along each arc
   keeps this fast on networks of up to about 300k arcs; then exact point-to-segment distance is
   computed. Why K=8: from `results/pipeline/candidate_miss_rate.csv` (train, 185,107 labelled
   points, miss = the true arc is not among the candidates):

   | K | miss rate (radius 75 m) |
   |---|---|
   | 1 | 44.6% |
   | 3 | 5.0% |
   | 5 | 2.1% |
   | **8** | **0.92%** |
   | 12 | 0.60% |
   | 16 | 0.48% |

   K=1 (just the nearest arc) misses 44.6% mostly because a two-way road gives two arcs at the
   same distance: distance alone cannot tell direction. K=16 only gains another 0.4% but doubles
   the table size, so we chose K=8. This choice used train only.
6. **Labels from the route** (`pathsnap/labels.py`). The route is an ordered arc list with far
   fewer rows than the track, so we align them with dynamic programming: each GPS point gets a
   route position, positions never go backwards, total point-to-arc distance is minimised.
   Points still more than **50 m** from their aligned arc are marked unlabelled and excluded from
   training and scoring. Evidence for 50 m: across train trips the median aligned distance is
   about 4.6 m and the 95th percentile about 14 m (median over trips of the per-trip values,
   `results/pipeline/trip_stats.csv`). Result: 185,107 of 208,246 kept points are labelled; 3 trips
   have label coverage below 90% (routes that do not cover the whole track).
7. **Candidate table.** One row per (GPS point, candidate arc) with a 0/1 label (1 if the arc is
   the aligned reference arc). Built by `python scripts/build_dataset.py`. Dataset size
   Result (`results/pipeline/dataset_summary.csv`, `results/pipeline/EXPERIMENTS.md`, entry E1):
train has 80 trips, 208,417 GPS points, 208,246 kept after cleaning, 185,107 labelled points and
1,241,235 candidate rows; 1,189,495 of those rows (labelled points only) are used for training.
The pooled candidate miss rate is 0.0092 on train. (`EXPERIMENTS.md` also lists 194,674 points that
have at least one candidate; the rest are removed points or points with no road within 75 m.)
student_test, only described and never used for a choice: 10 trips, 125,554 rows, 19,132
labelled points, miss rate 0.0025.

The route file is used only to make labels (and for the API's scoring). It is never a feature.

### 2.2.3 Missing-value analysis (partition: train)
Source: `results/eda/missing_values.csv`, run on all 80 train trips by `scripts/eda.py`.

| Check | Count |
|---|---|
| track rows with a non-finite value | 0 |
| track time steps with dt <= 0 | 0 |
| route arc ids outside the arcs file | 0 |
| nodes with a non-finite coordinate | 0 |
| arcs referencing a missing node | 0 |

So the raw files have no missing values. The data problems are not missing cells but messy
content, described next. Per-trip details are in `results/eda/trip_inventory.csv`.

### 2.2.4 What patterns we observed
Partition: train, 80 trips. Source: `results/eda/trip_inventory.csv` (summary in
`results/eda/EXPERIMENTS.md`, Run 1).
- Size: 208,417 GPS points, median 1,529 per trip (range 361 to 19,313). Road networks range
  from 649 to 307,392 arcs per trip, which is why candidate search needs a spatial index.
- Sampling: median time step is 1 s in every trip; 3 trips (2, 71, 97) have fractional timestamps.
- Gaps: 22 trips have a time step over 60 s (largest 153,064 s in trip 90; 52,810 s in trip 78).
  This is why we split trips into segments and never link points across a gap.
- Speed spikes: 10 trips have implied speeds above 70 m/s (max 180.4 m/s). These are GPS errors
  and motivate the speed filter in cleaning.
- Network quirks: 53 trips have duplicate arcs, 8 have self-loops, and the median trip has 20
  weakly connected components (only 1 trip is a single component). So the network is not one
  connected graph and the matcher must tolerate unreachable arcs.
- Reference routes: 10 trips have unconnected steps in the route (at most 3 per trip), so labels
  made from them can be slightly wrong; see label coverage in section 2.2.2.
What this told us to do: convert to metres, clean spikes and bad times, split at gaps, merge
duplicate arcs, use a KD-tree, and use heading as a feature to separate two-way roads.

### 2.2.5 Visualisations (partition for each must be stated in the caption)
All from the train partition (80 trips), made by `scripts/eda.py`.
- Sampling-interval histogram: `results/eda/sampling_hist.png`
- GPS trajectory plots: `results/eda/trajectories.png`
- Network-size plot (arcs per trip): `results/eda/network_size.png`
- Folium maps (GPS, nearby roads, reference route): `results/maps/eda_00000005.html`,
  `eda_00000013.html`, `eda_00000078.html`. `[[TODO: insert a screenshot of one of these]]`
Check each plot's real caption and axes before inserting; I have not viewed the images.

### 2.2.6 Feature engineering: what we extracted and why
All features are computed per (GPS point, candidate arc) from the track and the road network
only (`pathsnap/features.py`, list in `FEATURE_COLUMNS`). The same function builds the training
table and the live table inside the API, so training and serving see identical features.

| Feature | Meaning | Why we included it |
|---|---|---|
| `dist_m` | GPS point to arc distance (m) | The vehicle is usually closest to its own road. Strongest basic signal. |
| `dist_rank` | 0 = nearest candidate | Lets the model learn "nearest is right most of the time" without knowing absolute distances. |
| `dist_gap_to_best` | this distance minus the nearest candidate's | A 12 m candidate is fine if the nearest is 11 m but poor if the nearest is 2 m. |
| `heading_diff_deg` | angle between GPS travel direction and arc direction (0 to 180) | Separates the two arcs of a two-way road, which sit at the same distance. This is the fix for the K=1 failure above. |
| `heading_reliable` | 1 if the vehicle moved at least 2 m between neighbouring fixes | When stopped, GPS heading is pure noise; this flag tells the model to ignore the heading then. |
| `arc_len_m` | arc length | Very short arcs (junction pieces) behave differently from long roads. |
| `proj_t` | where the point projects along the arc (0 to 1) | Near the ends the point may really belong to the neighbouring arc. |
| `speed_mps` | speed from neighbouring fixes | Fast and slow movement behave differently (highway vs. junction). |
| `n_candidates` | number of candidates for the point | Dense city vs. sparse area. |
| `has_reverse_twin` | 1 if the opposite-direction arc is also a candidate | Tells the model heading is needed to decide. |
| `prev_dist_m`, `next_dist_m` | distance from the previous / next kept point to this same arc | A true arc stays close to the neighbours too; a wrong parallel road drifts away. A cheap bit of context without the HMM. |
| `conn_prev_nearest` | 1 if the arc equals or continues the previous point's nearest arc | Roads are connected; sudden jumps are suspicious. Uses no label. |

**How the features were selected.** The 13 features were designed from the geometry of the
problem (what distinguishes a right candidate from a wrong one), not picked by looking at test
results. Whether any feature was later dropped or added based on evidence is a separate claim:
No feature-importance or ablation file exists yet in `results/models/`, so as of now the
honest statement is: "features were chosen by reasoning, no ablation was run". If an ablation is
run later (CV on train only), replace this sentence with its result and file name.
Not used as features, by rule: the route, `true_arc`, `label`, anything from student_test or
evaluator trips.

---

## 3. Model experimentation and comparison

How models are evaluated (applies to all): the classifier scores each candidate; the
**observation-level Top-1 accuracy** picks the highest-scoring candidate per GPS point and
checks it against the reference arc. **CV Macro F1** is cross-validated on train with
**GroupKFold by `trip_id`** (whole trips are held out, never random rows, otherwise neighbouring
points of the same trip leak between folds). Macro F1 is on the candidate-level yes/no label,
which is imbalanced (most candidates are wrong), so it averages the F1 of both classes.
Log loss measures how honest the probabilities are, which matters because the HMM uses them.

### 3.1 Overall comparison table
Source for every cell: `results/models/model_comparison.csv`. Rows 1 and 3 are filled from
5-fold GroupKFold CV on train (untuned defaults, milestone 1). The "Top-1" and "Log loss" cells
are CV values until the one-time student_test run is done; the Student-test Macro F1 column and
rows 2, 4, 5 stay as placeholders. Route continuity here is computed on out-of-fold
classifier-only predictions (no HMM). Inference time is measured per 1,000 candidate rows.

| Model | Algorithm family | Key configuration | CV Macro F1 | Top-1 accuracy | Route continuity | Student-test Macro F1 | Log loss | Inference time |
|---|---|---|---|---|---|---|---|---|
| 1 Logistic Regression | linear / probabilistic | C=1.0, no class weight, standardised inputs | 0.8856 ± 0.0311 | 0.8372 ± 0.0569 (CV) | 0.9094 (out-of-fold) | `[[TODO: student_test, --final]]` | 0.1622 (CV) | 0.15 ms per 1,000 rows |
| 2 Random Forest | bagged trees | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` |
| 3 LightGBM (untuned) | boosted trees | 400 trees, learning rate 0.05, 31 leaves, min child samples 50, subsample 0.8, colsample 0.8 | 0.9105 ± 0.0324 | 0.8505 ± 0.0551 (CV) | 0.9287 (out-of-fold) | `[[TODO: student_test, --final]]` | 0.1121 (CV) | 3.77 ms per 1,000 rows |
| 4 MLP | neural network | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` |
| 5 KNN (distance-weighted) | instance-based | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` |

Baseline row to add: "nearest arc (K=1)" Top-1 = 1 minus 44.6% miss = about 55.4% on train
(`results/pipeline/candidate_miss_rate.csv`); this is the bar the models must beat.
If a metric does not apply to a model, write N/A and the reason in the cell. Model-selection
statement (which model, and why, in terms of CV on train, not student_test):
`[[TODO <- results/models/model_comparison.csv]]`.

### 3.2 Per-model subsections (repeat the block below for models 1 to 5)
Answer from the team's own experiments, not textbook text. Source folder: `results/models/`.

**Template (used for models 2, 4 and 5 until their results exist). Models 1 and 3 follow below.**

**Model N: `[[name]]`**
- **A1 Why suitable for PathSnap?** `[[TODO, in own words, tied to the 13 features]]`
- **A2 Strengths and limitations here:** `[[TODO, tied to observed results]]`
- **B5 Important hyperparameters and initial values:** `[[TODO <- artifacts / results/models/*params*]]`
- **B6 Loss / objective:** `[[TODO, e.g. log loss for logistic regression; state what was actually used]]`
- **C7 Performance on student_test:** `[[TODO <- results/models/model_comparison.csv]]`
- **C8 Observation-level Top-1, Macro F1, Log loss on student_test:** `[[TODO]]`
- **C9 Over/underfitting and the evidence:** `[[TODO: train score vs CV score gap, learning curve or loss curve <- results/models/]]`

**Model 1: Logistic Regression** (`pathsnap/models.py`, results `results/models/model_comparison.csv`, `cv_folds.csv`)
- **A1 Why suitable?** It is the simplest model that outputs a probability for "is this candidate the true arc?", so it is a fair baseline and its probabilities work as HMM emissions. Inputs are standardised first because features have different units (metres, degrees, 0 to 1).
- **A2 Strengths and limitations (from our runs):** It is fast (about 1.7 s per fold to fit) and train and validation scores are almost the same, so it is stable. But it can only draw a linear boundary, so it cannot express rules like "close AND heading matches", and it trails LightGBM on every metric in every fold (see below).
- **B5 Hyperparameters:** C=1.0 (inverse regularisation strength), class_weight=None, solver lbfgs, max_iter 1000, seed 42. No tuning done yet.
- **B6 Objective:** log loss (cross-entropy) with L2 regularisation.
- **C7 to C8 Student-test results:** `[[TODO: Top-1, Macro F1, log loss on student_test <- results/models/final_student_test.csv, produced once at the end]]`
- **C9 Over/underfitting evidence** (`cv_folds.csv`, mean over 5 folds): train Top-1 0.8376 vs validation 0.8372; train Macro F1 0.8869 vs validation 0.8856; train log loss 0.1596 vs validation 0.1622. The train and validation values are almost identical, so there is no overfitting. Scores are not high either, and the model is beaten by LightGBM, which suggests it is limited by its linear form (mild underfitting). The large spread across folds (validation Top-1 from 0.771 in fold 0 to 0.912 in fold 2) shows that results depend on which trips are held out, not on overfitting.

**Model 3: LightGBM, untuned** (same files)
- **A1 Why suitable?** The tabular features interact (a close arc with the wrong heading is wrong; a farther arc with the right heading and continuity may be right). Boosted trees learn such rules automatically, need no scaling and train quickly on 1.19 million rows.
- **A2 Strengths and limitations (from our runs):** Best CV scores so far on all metrics. Costs more: about 9.7 s per fold to fit and 3.77 ms per 1,000 rows to predict (25 times slower than Logistic Regression, still small). It overfits somewhat (below).
- **B5 Hyperparameters and initial values:** n_estimators 400, learning_rate 0.05, num_leaves 31, min_child_samples 50, subsample 0.8, colsample_bytree 0.8, reg_lambda 1.0, seed 42. Tuning: see section 4.
- **B6 Objective:** binary log loss, optimised by gradient boosting.
- **C7 to C8 Student-test results:** `[[TODO <- results/models/final_student_test.csv]]`
- **C9 Over/underfitting evidence** (`cv_folds.csv`, mean over 5 folds): train Top-1 0.8788 vs validation 0.8505; train Macro F1 0.9281 vs validation 0.9105; train log loss 0.0913 vs validation 0.1121. Training is clearly better than validation on all three, so there is a mild overfit. The per-iteration validation loss curve is `results/models/plots/loss_curve_lightgbm.png`: What the plot shows (CV fold 0 only, not an average): validation log loss falls fast over the first about 60 iterations, then flattens at about 0.155 from roughly iteration 150 onward, with at most a very slight rise near iteration 400, while training loss keeps falling to about 0.083. The widening gap between the two curves is the overfitting; extra trees past about 150 add almost nothing on validation, so fewer trees or stronger regularisation are reasonable things for the tuning step to test. (Fold 0 values match `cv_folds.csv`: train 0.0829, validation 0.1549.)

**Comparison of models 1 and 3 on CV (`results/models/model_comparison.csv`, `cv_folds.csv`):**
LightGBM beats Logistic Regression in all five folds on both Top-1 and Macro F1. Mean CV Top-1 is
0.8505 vs 0.8372, a gap of about 0.013, which is much smaller than the spread between folds
(standard deviation about 0.055). Because folds are different trips, compare per fold, not just
means; the per-fold ordering is consistent but the gap in absolute terms is modest. Both are
well above the nearest-arc baseline of about 0.554 Top-1 (1 minus the K=1 miss rate of 0.446).

### 3.3 Model visualisations (each must come from a real run)
- CV Macro F1: `results/models/plots/cv_f1_macro.png` (also `cv_top1.png`). Currently models 1 and 3 only; regenerate when all five are done.
- Student-test Top-1 and Macro F1 `[[TODO]]`
- Log loss comparison: `results/models/plots/cv_log_loss.png` (CV values for now).
- Confusion matrices (optional) `[[TODO]]`
- Training/validation loss curves: **only** MLP and boosting with an evaluation set. Do not
  draw curves for Logistic Regression (if solved in closed form/iterations without epoch
  history), Random Forest or KNN. LightGBM: `results/models/plots/loss_curve_lightgbm.png` (exists). MLP: `[[TODO]]`.

---

## 4. Bayesian hyperparameter optimisation (chosen model: `[[TODO]]`)
Tool: Optuna (TPE sampler), seed `config.SEED`. Every trial is saved to CSV.
1. Hyperparameters searched: `[[TODO <- scripts/bayes_opt.py]]`
2. Ranges / distributions: `[[TODO]]`
3. Objective optimised: `[[TODO, expected: CV Macro F1 or CV log loss on train, GroupKFold by trip_id]]`
4. Number of trials: `[[TODO <- results/models/optuna_trials.csv]]`
5. Best values: `[[TODO]]`
6. Did tuning beat untuned? `[[TODO]]` Untuned vs tuned, same data and metrics:

| | CV Macro F1 | Student-test Top-1 | Student-test Macro F1 | Log loss | Inference time |
|---|---|---|---|---|---|
| Untuned | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` |
| Tuned | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` |

Plots: optimisation history, parameter importance `[[TODO: results/models/plots/]]`.
Be honest if the gain is small or within noise.

---

## 5. HMM and Viterbi route reconstruction
Plain-language idea: the classifier judges each GPS point alone and can jump between roads, or
pick the wrong direction of a two-way road. A real vehicle cannot teleport, so we look for the
single best *whole* chain of arcs. Viterbi is the dynamic-programming method that finds that
chain efficiently instead of trying every combination. Code: `pathsnap/hmm.py`,
`pathsnap/pipeline.py`, `scripts/hmm_eval.py`. Details also in `report/hmm_and_data_pipeline.md`.
All numbers: `results/pipeline/hmm_comparison.csv`, `hmm_params.json`, `hmm_grid.csv`,
`hmm_per_trip_*.csv`.

1. **Hidden states:** for each GPS point, one of its (up to 8) candidate arcs.
2. **Emission:** log of the classifier probability that the candidate is correct, multiplied by an
   emission weight. The no-ML baseline uses a Gaussian on the point-to-arc distance (sigma 10 m).
   Probabilities are clipped to at least 1e-6 so the log is finite.
3. **Transition:** from candidate a at one point to candidate b at the next, score
   exp(-|road distance - GPS distance| / beta). Road distance is the shortest path on the arc
   graph (bounded Dijkstra search) between the projections of the two points. A car moves about as
   far on the road as the GPS says it moved, so large disagreement is penalised.
4. **Gaps and missing candidates:** the trip is cut wherever the time gap is over 60 s and each
   piece is decoded separately (no linking across a gap). Points with no candidate get no arc
   (-1). Network search is bounded, so far-apart points do not trigger a huge search.
5. **Disconnected consecutive observations:** if b cannot be reached from a, the transition gets a
   fixed penalty (the "floor", in log units), never zero probability, so one bad point cannot kill
   the whole path. Viterbi runs in log space to avoid underflow.
6. **Tuned values** (4 parameters, chosen on train out-of-fold probabilities, GroupKFold by trip;
   120 grid trials per method in `hmm_grid.csv`): classifier+HMM uses beta 10, floor 25, emission
   weight 0.1; distance+HMM uses beta 50, floor 50, emission weight 0.25 (`hmm_params.json`).
   Caveat: these were tuned and scored on the same 80 train trips, so the HMM numbers are slightly
   optimistic. The top few grid settings are within about 0.001 Top-1 of each other, so the result
   is not sensitive to the exact values. The honest check is the one-time student_test run.

**Did HMM decoding improve the final route?** Train, 80 trips, 185,107 labelled points, out-of-fold
LightGBM probabilities (pooled over points; `hmm_comparison.csv`):

| Method | Top-1 | Route continuity | Student-test Top-1 | Student-test continuity |
|---|---|---|---|---|
| Nearest arc (distance only) | 0.5536 | 0.8243 | `[[TODO --final]]` | `[[TODO --final]]` |
| Distance emission + HMM (no ML) | 0.9289 | 0.9813 | `[[TODO --final]]` | `[[TODO --final]]` |
| Classifier only (LightGBM) | 0.8502 | 0.9442 | `[[TODO --final]]` | `[[TODO --final]]` |
| Classifier + HMM | 0.9390 | 0.9809 | `[[TODO --final]]` | `[[TODO --final]]` |

What this shows, stated honestly:
- **Most of the gain comes from the HMM, not the classifier.** The HMM lifts the classifier from
  0.850 to 0.939 Top-1 (+8.9 points) and route continuity from 0.944 to 0.981. A distance-only HMM
  with no machine learning at all already reaches 0.929 Top-1 and 0.981 continuity.
- **The classifier adds about 1 point on top of the HMM** (0.939 vs 0.929 Top-1), and no gain in
  continuity (0.9809 vs 0.9813). The tuned emission weight is small (0.1), meaning the HMM gives
  the classifier's opinion little weight relative to the road-connection model. We should not
  overstate the classifier's role: the work that matters most here is candidate generation,
  heading features (which help the classifier itself) and the transition model.
- **Per trip (80 trips):** classifier+HMM beats classifier-only on 79 trips and is worse on one.
  Biggest gains: trip 2 (0.584 to 0.891), trip 8 (0.603 to 0.863), trip 33 (0.476 to 0.724),
  trip 21 (0.776 to 0.994). Median gain per trip is about 4.7 points. Classifier+HMM beats
  distance+HMM on 61 trips and is worse on 17.
- **Where HMM did not help:** trip 63 got worse with the HMM (0.846 to 0.829), and on 17 trips the
  distance-only HMM was better than classifier+HMM, so the classifier's probabilities sometimes
  pull the path the wrong way. `[[TODO: look at trip 63 (and trip 33, still only 0.724) on a Folium
  map and state the cause only after seeing it. Do not guess.]]`
- Unlabelled points (about 11% of kept points, section 2.2.2) are not scored.

Folium map showing classifier-only vs classifier+HMM on one trip: `[[TODO: results/maps/trip_<id>.html
and screenshot]]`.

---

## 6. Geographical visualisation (Folium)
Maps are produced by `pathsnap.viz.trip_map` (report maps: GPS, nearby roads, reference route,
predicted route, with legend and layer control) and `ui/map_view.make_result_map` (UI: GPS and
predicted route only; the reference route is never shown there).
Maps present in the repo now: `results/maps/trip_00000005.html`, `trip_00000039.html`,
`trip_00000046.html`. Add at least one screenshot: `[[TODO: screenshot of results/maps/trip_<id>.html,
state partition]]`. Include one good case and one failure case.

## 7. API, UI and deployment
- Architecture: FastAPI on Laptop 1 (`api/main.py`: `/health`, `/trips`, `/predict`), Streamlit UI
  on Laptop 2 (`ui/app.py`), reached over the same network. The API runs the full pipeline live
  on the chosen trip; the reference route is read only to compute the accuracy and is never sent
  to the UI.
- Screenshots of the working UI `[[TODO: screenshots]]`
- Screenshots of API responses in the receiving laptop's command window `[[TODO]]`
- Run instructions: see `README.md` `[[TODO: confirm it matches what was demonstrated]]`.
The submitted code must be identical to what is demonstrated.

## 8. Academic integrity and AI use
- Evaluator trips were never used for training, feature choice, tuning or thresholds.
- Student_test was not used for any decision; its numbers were computed once at the end.
- Cross-validation is GroupKFold by trip; no random splits.
- Every number in this report traces to a file in `results/`.
- AI assistance is disclosed in full in `report/ai_use_disclosure.md`.
