# PathSnap technical report (team skeleton)

> **Status: skeleton.** Layout follows `Deliverables.pdf` sections 2.2 to 8. Every number is
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
   `[[TODO: rows, points, positive rate <- results/pipeline/dataset_summary.csv]]`.

The route file is used only to make labels (and for the API's scoring). It is never a feature.

### 2.2.3 Missing-value analysis (partition: train)
`[[TODO: table of missing/non-finite values per file type and per trip <- results/eda/trip_inventory.csv, results/eda/missing_values.csv]]`
Known data facts to confirm against that table (from the team's check of the raw files, not
yet our own EDA output): some trips have fractional timestamps, large time gaps, speed spikes
and some reference routes contain unconnected steps.

### 2.2.4 What patterns we observed
`[[TODO: sampling-interval histogram, gaps, speed spikes, network size range, duplicate arcs,
self-loops, connected components <- results/eda/trip_inventory.csv + plots in results/eda/]]`

### 2.2.5 Visualisations (partition for each must be stated in the caption)
- GPS trajectory plots `[[TODO: results/eda/*.png]]`
- Road-network visualisation `[[TODO: results/eda/*.png]]`
- Folium map of a chosen trip `[[TODO: results/maps/eda_<trip>.html + screenshot]]`

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
`[[TODO: feature-importance / ablation result (CV on train only) <- results/models/feature_importance.csv, results/models/ablation.csv]]`.
If no ablation was run, write exactly that: "features were chosen by reasoning, no ablation run".
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
Source for every cell: `results/models/model_comparison.csv`.

| Model | Algorithm family | Key configuration | CV Macro F1 | Top-1 accuracy | Route continuity | Student-test Macro F1 | Log loss | Inference time |
|---|---|---|---|---|---|---|---|---|
| 1 Logistic Regression | linear | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` |
| 2 Random Forest | bagged trees | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` |
| 3 LightGBM | boosted trees | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` |
| 4 MLP | neural network | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` |
| 5 KNN (distance-weighted) | instance-based | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` |

Baseline row to add: "nearest arc (K=1)" Top-1 = 1 minus 44.6% miss = about 55.4% on train
(`results/pipeline/candidate_miss_rate.csv`); this is the bar the models must beat.
If a metric does not apply to a model, write N/A and the reason in the cell. Model-selection
statement (which model, and why, in terms of CV on train, not student_test):
`[[TODO <- results/models/model_comparison.csv]]`.

### 3.2 Per-model subsections (repeat the block below for models 1 to 5)
Answer from the team's own experiments, not textbook text. Source folder: `results/models/`.

**Model N: `[[name]]`**
- **A1 Why suitable for PathSnap?** `[[TODO, in own words, tied to the 13 features]]`
- **A2 Strengths and limitations here:** `[[TODO, tied to observed results]]`
- **B5 Important hyperparameters and initial values:** `[[TODO <- artifacts / results/models/*params*]]`
- **B6 Loss / objective:** `[[TODO, e.g. log loss for logistic regression; state what was actually used]]`
- **C7 Performance on student_test:** `[[TODO <- results/models/model_comparison.csv]]`
- **C8 Observation-level Top-1, Macro F1, Log loss on student_test:** `[[TODO]]`
- **C9 Over/underfitting and the evidence:** `[[TODO: train score vs CV score gap, learning curve or loss curve <- results/models/]]`

### 3.3 Model visualisations (each must come from a real run)
- CV Macro F1, all models `[[TODO: results/models/plots/]]`
- Student-test Top-1 and Macro F1 `[[TODO]]`
- Log loss comparison `[[TODO]]`
- Confusion matrices (optional) `[[TODO]]`
- Training/validation loss curves: **only** MLP and boosting with an evaluation set. Do not
  draw curves for Logistic Regression (if solved in closed form/iterations without epoch
  history), Random Forest or KNN. `[[TODO: results/models/plots/loss_*.png]]`

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
Plain-language idea: the classifier judges each GPS point alone and can jump between roads.
A real vehicle cannot teleport, so we look for the single best *whole* chain of arcs. Viterbi is
the dynamic-programming method that finds that chain efficiently instead of trying every
combination.

(The design below is from the team's agreed plan in `pathsnap/hmm.py` and `docs/CONTRACTS.md`.
At the time of writing the decoder was not yet implemented, so confirm each statement against
the final `hmm.py` before submitting and fix any difference.)

1. **Hidden states:** for each GPS point, one of its candidate arcs.
2. **Emission:** the classifier's probability for that candidate (log). Baseline without ML:
   Gaussian on `dist_m`.
3. **Transition:** high if the next arc is the same or directly connected (`arcs[a,1] == arcs[b,0]`);
   otherwise scored from the difference between road distance and straight GPS distance, with a
   small floor so no path is ever impossible. Parameters `[[TODO <- results/pipeline/hmm_params.json]]`.
4. **Gaps / missing candidates:** the trip is split at time gaps over 60 s (segments from
   `clean_track`) and each part is decoded separately. Points without candidates get no arc.
   `[[TODO: confirm]]`
5. **Disconnected consecutive observations:** `[[TODO: describe floor behaviour as implemented]]`
6. **Did HMM help?** Comparison, train (out-of-fold probabilities) and student_test once:

| Method | Top-1 | Route continuity |
|---|---|---|
| Nearest arc only / distance-only HMM | `[[TODO]]` | `[[TODO]]` |
| Classifier only | `[[TODO]]` | `[[TODO]]` |
| Classifier + HMM/Viterbi | `[[TODO]]` | `[[TODO]]` |

Source: `results/pipeline/hmm_comparison.csv`. Discuss trips where HMM helped and where it did
not (`[[TODO: per-trip examples from the same file and a Folium map]]`).

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
