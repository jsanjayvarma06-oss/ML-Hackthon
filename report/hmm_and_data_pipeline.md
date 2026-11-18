# HMM route reconstruction (Sanjay's report section, HMM part)

Every number below comes from `results/pipeline/` (see `EXPERIMENTS.md` there).

## Why an HMM
Picking the best-scoring candidate for each GPS point on its own ignores that a car drives
along connected roads. Two-way roads make this worse: both directions of a road are the same
distance from the GPS point, so a per-point choice is often the wrong direction.

## Model
- Hidden state at each point: one of its (up to 8) candidate arcs.
- Emission: log of the classifier probability for the candidate. Baseline without ML: Gaussian on the distance to the arc.
- Transition: exp(-|network distance - GPS distance| / beta). Network distance is the
  shortest path on the arc graph (bounded Dijkstra) between the projections of the two points
  on their arcs. Unreachable pairs get a fixed floor penalty, never zero probability.
- Viterbi in log space; the trip is cut wherever the time gap is over 60 s.
- Tunable: beta, floor, emission weight (4 numbers). Chosen on train only, see `hmm_grid.csv`.

## Results (train, 80 trips, out-of-fold LightGBM probabilities)
Source: `results/pipeline/hmm_comparison.csv`.

| Method | Top-1 | Route continuity |
|---|---|---|
| Nearest arc (distance only) | 0.5536 | 0.8243 |
| Distance emission + HMM | 0.9289 | 0.9813 |
| Classifier only | 0.8502 | 0.9442 |
| Classifier + HMM | 0.9390 | 0.9809 |

Reading: the HMM is worth about +9 points top-1 on top of the classifier and +37 on top of
nearest arc. The classifier adds about +1 point on top of a distance-only HMM, so most of the
gain comes from the transition model. Student_test numbers are added once at the end
(`results/pipeline/hmm_comparison_student_test.csv`).
