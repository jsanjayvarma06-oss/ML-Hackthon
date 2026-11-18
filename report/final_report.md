# PathSnap technical report (team skeleton)

> **Status: filled from train results (data, EDA, all 5 models, tuning, HMM). Student-test cells are placeholders.** Layout follows `Deliverables.pdf` sections 2.2 to 8. Every number is
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
Source for every filled cell: `results/models/model_comparison.csv` (5-fold GroupKFold CV on train,
untuned defaults, same folds for every model). Top-1 is mean over folds ± standard deviation. Route
continuity is computed on out-of-fold classifier-only predictions (no HMM). Inference time is per
1,000 candidate rows. The Student-test Macro F1 column is a placeholder: the one-time student_test
run has not been done and waits for the team's go-ahead.

| Model | Algorithm family | Key configuration | CV Macro F1 | CV Top-1 | Route continuity | Student-test Macro F1 | CV log loss | Inference time |
|---|---|---|---|---|---|---|---|---|
| 1 Logistic Regression | linear / probabilistic | C=1.0, standardised inputs | 0.8856 | 0.8372 ± 0.0569 | 0.9094 | `[[TODO: --final]]` | 0.1622 | 0.15 ms |
| 2 Random Forest | bagging ensemble | 150 trees, max depth 14, min leaf 20, max features sqrt | 0.9092 | 0.8480 ± 0.0559 | 0.9209 | `[[TODO: --final]]` | 0.1157 | 2.09 ms |
| 3 LightGBM | boosting | 400 trees, lr 0.05, 31 leaves, min child 50 | 0.9105 | 0.8505 ± 0.0551 | 0.9287 | `[[TODO: --final]]` | 0.1121 | 3.77 ms |
| 4 MLP | neural network | hidden (64, 32), alpha 1e-4, trained on 100,000 points per fit | 0.9071 | 0.8451 ± 0.0563 | 0.9276 | `[[TODO: --final]]` | 0.1190 | 0.32 ms |
| 5 KNN (distance-weighted) | instance-based | 25 neighbours, trained on 40,000 points per fit | 0.9021 | 0.8379 ± 0.0549 | 0.9191 | `[[TODO: --final]]` | 0.2536 | 312.8 ms |
| Baseline: nearest arc | none | K=1 | N/A, not a probabilistic classifier | about 0.554 (1 minus K=1 miss rate 0.446) | 0.8243 | `[[TODO: --final]]` | N/A | N/A |

The tuned LightGBM is reported in section 4. Student-test Top-1 and log loss will be added from
`results/models/final_student_test.csv` when that run is done.

**Model selection (decided on CV only, before any student_test run).** Rule written before tuning
in `scripts/bayes_opt.py`: highest CV Top-1, ties within 0.002 broken by lower log loss.
LightGBM has both the highest Top-1 (0.8505) and the lowest log loss (0.1121); runner-up Random
Forest is 0.0025 behind. **Be honest about the margin:** the fold-to-fold standard deviation is
about 0.055 for every model, because some validation folds contain hard trips, while the gap between
best and worst model is only 0.013. Models are ranked on identical folds so the ordering is fair,
but differences of a few thousandths should not be read as significant. LogReg and KNN are clearly
behind on log loss; the three non-linear models are very close.

### 3.2 Per-model subsections
Sources: `results/models/model_comparison.csv`, `cv_folds.csv`, `report/models_and_tuning.md`.
Student-test items (C7, C8) are placeholders until the one-time run. Overfitting evidence below is
mean train vs validation over the 5 folds (`cv_folds.csv`).

**Model 1: Logistic Regression**
- **A1 Why suitable?** Simplest model that outputs a probability for "is this candidate the true arc?"; a fair baseline, and probabilities usable as HMM emissions. Inputs are standardised because features have different units.
- **A2 Strengths / limitations:** Fast (1.7 s per fold) and stable, but linear, so it cannot express "close AND same direction". It trails the non-linear models on every metric.
- **B5 Hyperparameters:** C=1.0, no class weight, lbfgs, max_iter 1000, seed 42. **B6 Objective:** log loss with L2 regularisation.
- **C7 to C8 Student-test:** `[[TODO: --final]]`
- **C9 Fit:** train Top-1 0.8376 vs validation 0.8372; train log loss 0.1596 vs 0.1622. No overfitting; the model is limited by its linear form (mild underfitting).

**Model 2: Random Forest**
- **A1 Why suitable?** Many decorrelated trees average out noise and capture non-linear rules without scaling.
- **A2 Strengths / limitations:** Close to LightGBM (Top-1 within 0.003) but about 4 times slower to fit (37 s per fold).
- **B5 Hyperparameters:** 150 trees, max depth 14, min samples per leaf 20, max features sqrt, max_samples 0.5, seed 42. **B6 Objective:** Gini impurity per tree; probabilities from tree voting.
- **C7 to C8 Student-test:** `[[TODO: --final]]`
- **C9 Fit:** train Top-1 0.8622 vs validation 0.8480; train log loss 0.1021 vs 0.1157. Mild overfitting, limited by the depth and leaf-size settings.

**Model 3: LightGBM (untuned)**
- **A1 Why suitable?** Features interact (a close arc with the wrong heading is wrong); boosted trees learn such rules, need no scaling and are fast on 1.19 million rows.
- **A2 Strengths / limitations:** Best CV Top-1 and log loss. 9.7 s per fold to fit, 3.77 ms per 1,000 rows to predict. Mild overfitting.
- **B5 Hyperparameters:** 400 trees, lr 0.05, 31 leaves, min child samples 50, subsample 0.8, colsample 0.8, reg_lambda 1.0, seed 42. **B6 Objective:** binary log loss via gradient boosting.
- **C7 to C8 Student-test:** `[[TODO: --final]]`
- **C9 Fit:** train Top-1 0.8788 vs validation 0.8505; train log loss 0.0913 vs 0.1121 (mild overfit). Loss curve (CV fold 0 only, `plots/loss_curve_lightgbm.png`): validation loss flattens near 0.155 from about iteration 150 while training loss keeps falling to about 0.083.

**Model 4: MLP (neural network)**
- **A1 Why suitable?** A neural-network family member that can learn non-linear combinations of the 13 features and has a real epoch history.
- **A2 Strengths / limitations:** Competitive (Top-1 0.8451) and fast at prediction (0.32 ms per 1,000 rows), but not better than boosting on tabular features. Trained on a fixed 100,000-point subsample per fit for compute reasons (validation folds are never subsampled).
- **B5 Hyperparameters:** hidden layers (64, 32), alpha (L2) 1e-4, constant learning rate, early stopping, seed 42. **B6 Objective:** log loss (cross-entropy), optimised by gradient descent.
- **C7 to C8 Student-test:** `[[TODO: --final]]`
- **C9 Fit:** train Top-1 0.8563 vs validation 0.8451; train log loss 0.1082 vs 0.1190. Small gap, mild overfit. Loss curve: `plots/loss_curve_mlp.png` (CV fold 0). Training loss drops quickly in the first 2 epochs then flattens at about 0.10 over about 27 epochs. The orange line in that plot is the MLP's validation **score** (an accuracy-type value near 0.95), not a loss, so it must not be read as a loss curve. `[[TODO: ask Kushal to relabel or split this axis before it goes in the PDF]]`

**Model 5: KNN (distance-weighted)**
- **A1 Why suitable?** An instance-based method with no training assumptions; included as a different family for comparison.
- **A2 Strengths / limitations:** Cheapest to fit (0.4 s) but the slowest to predict (312.8 ms per 1,000 rows) and its probabilities are poorly calibrated, which makes it a weak HMM emission. Trained on a fixed 40,000-point subsample per fit.
- **B5 Hyperparameters:** 25 neighbours, distance weighting, standardised inputs. **B6 Objective:** none (no training loss); it votes among neighbours.
- **C7 to C8 Student-test:** `[[TODO: --final]]`
- **C9 Fit:** strong overfitting: train Top-1 0.9908 and train log loss 0.0000 (each training point is its own nearest neighbour) against validation Top-1 0.8379 and log loss 0.2536, the worst of all models.

### 3.3 Model visualisations (all from real runs)
- CV Macro F1, all five models: `results/models/plots/cv_f1_macro.png`; CV Top-1: `cv_top1.png`.
- Log loss comparison: `results/models/plots/cv_log_loss.png` (CV values).
- Student-test Top-1 and Macro F1 comparison, confusion matrices: `[[TODO: after the --final run]]`.
- Loss curves, only for models with real iteration histories: LightGBM `plots/loss_curve_lightgbm.png` (and `loss_curve_lightgbm_tuned.png`), MLP `plots/loss_curve_mlp.png`. None for Logistic Regression, Random Forest or KNN.

---

## 4. Bayesian hyperparameter optimisation (chosen model: LightGBM)
Sources: `results/models/optuna_trials.csv`, `best_params.json`, `tuned_vs_untuned.csv`,
`cv_folds_tuned.csv`, `plots/optuna_history.png`, `plots/optuna_param_importance.png`.
Tool: Optuna with the TPE sampler (Tree-structured Parzen Estimator), seed 42. In simple terms,
TPE keeps track of which parameter regions scored well versus badly and proposes new settings
where "good" is most likely, so it needs far fewer trials than a grid search.
1. **Hyperparameters searched:** n_estimators, learning_rate, num_leaves, min_child_samples, subsample, colsample_bytree, reg_lambda.
2. **Ranges:** n_estimators 100 to 600; learning_rate 0.01 to 0.2 (log scale); num_leaves 8 to 128 (log); min_child_samples 10 to 400 (log); subsample 0.5 to 1; colsample_bytree 0.5 to 1; reg_lambda 0.001 to 30 (log). (From `report/models_and_tuning.md`; confirm against `scripts/bayes_opt.py`.)
3. **Objective:** mean validation Top-1 over GroupKFold(3) by trip on the 80 train trips (3 folds so a trial costs about 30 s). Why LightGBM: it won the rule on CV Top-1 and log loss (section 3.1).
4. **Trials:** 40, every trial saved in `optuna_trials.csv` with parameters, per-fold score and duration. Trial scores: min 0.8336, median 0.8492, max 0.8496 (best is trial 33).
5. **Best values:** n_estimators 450, learning_rate 0.0326, num_leaves 50, min_child_samples 121, subsample 0.592, colsample_bytree 0.572, reg_lambda 0.197.
6. **Did tuning beat untuned?** Barely. Same 5 folds, same data (`tuned_vs_untuned.csv`):

| | CV Macro F1 | CV Top-1 | CV log loss | Train Top-1 | Route continuity (OOF) | Student-test Top-1 | Student-test Macro F1 | Inference time |
|---|---|---|---|---|---|---|---|---|
| Untuned | 0.9105 | 0.8505 | 0.1121 | 0.8788 | 0.9287 | `[[TODO: --final]]` | `[[TODO: --final]]` | 3.77 ms / 1,000 rows |
| Tuned | 0.9111 | 0.8510 | 0.1116 | 0.8762 | 0.9282 | `[[TODO: --final]]` | `[[TODO: --final]]` | 5.33 ms / 1,000 rows (`cv_folds_tuned.csv`; machine load may differ) |

**Plain reading:** tuning improved CV Top-1 by only +0.0005, far inside the fold spread (about
0.055), so it is not a meaningful gain. Most trials landed near 0.849, which suggests the default
LightGBM was already near the plateau for these 13 features. The visible benefit is a slightly
smaller train-validation gap (train Top-1 0.8788 to 0.8762). Bigger improvements are more likely to
come from features or the HMM than from hyperparameters. Caveat: tuning and the comparison used the
same 80 train trips and no nested cross-validation, so the tuned figure is mildly optimistic.
The model chosen for the system is `lightgbm_tuned`.

---

## 5. HMM and Viterbi route reconstruction
> **Tuned-model update (from the lead thread's re-run, `results/pipeline/EXPERIMENTS.md` entry E3; `[[TODO: verify these two rows against that file once it is applied]]`):** with out-of-fold probabilities from the tuned LightGBM, classifier-only is 0.8508 Top-1 / 0.9441 continuity and classifier + HMM is 0.9388 / 0.9809, with unchanged HMM parameters (beta 10, floor 25, weight 0.1). The tables below show the untuned-model run in `hmm_comparison.csv`; use the tuned numbers for the final model.
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
- Model picker and comparison page: the UI lets the evaluator choose which trained model runs and has a "Model comparison" page. The API serves them through `GET /models` and `GET /comparison` (as reported by the lead thread; `[[TODO: confirm against api/main.py and ui/app.py on the latest branch KushalJayavarapu/ML-Hackthon-kushal, claude/project-thread-0jgx9i]]`).
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
