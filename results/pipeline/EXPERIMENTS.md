# Experiment log: data pipeline (owner: Sanjay)

Each entry: command, code commit, date, outputs, headline numbers. Numbers are copied from the output files.

## E1. Build candidate + feature tables (features v1)
- Command: `python scripts/build_dataset.py`
- Commit: features v1 (`Add candidate features v1 and dataset builder`), base main @ 01ebede
- Date: 2026-11-18
- Outputs: `data/processed/train.parquet`, `data/processed/student_test.parquet` (gitignored, rebuilt by the command), `results/pipeline/dataset_summary.csv`
- Settings: K=8 candidates within 75 m, gap split 60 s, label cut-off 50 m, 13 features (`pathsnap.features.FEATURE_COLUMNS`)
- Headline: train 80 trips, 208,417 GPS points, 208,246 kept after cleaning, 194,674 with at least one candidate (1,241,235 rows), 185,107 labelled points, pooled candidate miss rate 0.0092. student_test (descriptive stats only, not used for any choice): 10 trips, 125,554 rows, 19,132 labelled points, miss rate 0.0025.


## E2: HMM parameter search on train (OOF LightGBM probabilities)
- Command: `python scripts/hmm_eval.py --oof results/models/oof_lightgbm.parquet`
  (the OOF file used was `oof_lightgbm.parquet` from the models thread, 5-fold GroupKFold by trip, untuned LightGBM)
- Code: features v1 patch + `pathsnap/hmm.py`, `pathsnap/pipeline.py`, `scripts/hmm_eval.py` (this change)
- Data: 80 train trips, 185,107 labelled points. student_test not used.
- Outputs: `results/pipeline/hmm_grid.csv` (every grid trial), `hmm_comparison.csv`,
  `hmm_per_trip_<method>.csv`, `hmm_params.json`
- Grid: beta {1,2,5,10,20,50} m, floor {25,50,100,200} nats, emission weight
  {0.05..1} (classifier) or {0.25..4} (distance). An earlier narrower grid hit its edge, so it was widened.
- Headline (pooled over labelled points, top-1 / route continuity):
  distance-only 0.5536 / 0.8243; distance + HMM 0.9289 / 0.9813;
  classifier-only 0.8502 / 0.9442; classifier + HMM 0.9390 / 0.9809.
- Chosen classifier_hmm parameters: beta 10, floor 25, weight 0.1 (many other settings
  within 0.001 top-1, see hmm_grid.csv).
- Caveats: 4 HMM numbers were picked on the same train trips they are scored on (small
  optimism). Unlabelled points have no OOF probability and get a uniform probability.
  The OOF file has no rows for unlabelled points, so they are context only, never scored.

## Run 2: full pipeline smoke test
- `predict_trip` with distance-only emission and with a LightGBM bundle fitted on all train
  (temporary, in-sample, only to check that nothing crashes) ran on all 80 train trips without error.
  Slowest trips ~2.2 s end to end. No accuracy was recorded from this run.

## E3: HMM parameter search on the tuned LightGBM OOF probabilities
- Command: `python scripts/hmm_eval.py --oof results/models/oof_lightgbm_tuned.parquet` (2 min 21 s)
- OOF file: tuned LightGBM (best_params.json), GroupKFold(5) by trip, from `scripts/bayes_opt.py` (models thread)
- Data: 80 train trips, 185,107 labelled points. student_test not used.
- Outputs: `hmm_grid.csv`, `hmm_comparison.csv`, `hmm_per_trip_*.csv`, `hmm_params.json` (overwrote E2's)
- Headline (pooled Top-1 / route continuity): classifier-only 0.8508 / 0.9441; classifier + HMM 0.9388 / 0.9809;
  distance + HMM unchanged 0.9289 / 0.9813. Chosen params unchanged: beta 10, floor 25, emission weight 0.1.
- Reading: tuning the classifier changed classifier-only Top-1 by +0.0006 and classifier+HMM by −0.0002 (0.9390 in E2), i.e. no measurable difference.
- Then `python scripts/export_model.py` wrote `artifacts/model.joblib` = lightgbm_tuned with these hmm_params.

## FINAL: one-time student_test HMM evaluation (approved by Wolf)
- Command: `python scripts/hmm_eval.py --final` at commit c804a3b, model artifacts/model.joblib (lightgbm_tuned), params from hmm_params.json (E3)
- Output: `hmm_comparison_student_test.csv` (10 trips, 19,132 labelled points)
- Headline Top-1 / route continuity: distance only 0.5351 / 0.8209; distance + HMM 0.9368 / 0.9809; classifier only 0.8630 / 0.9431; classifier + HMM 0.9214 / 0.9818
- Reading (honest): on student_test the no-ML decoder (distance + HMM) beats classifier + HMM by 1.5 points, the reverse of train CV (0.9289 vs 0.9388). We do NOT switch models because of this (student_test must not drive choices); the report discusses it as a limitation: the HMM does most of the work, and the classifier's extra signal did not generalise to these 10 trips.
- Not used for any decision.

## E4: push for higher Top-1 after the final run (team asked for >= 95%), train only
Context: after the student_test run the team asked for at least 95% accuracy. No such threshold is in the five official PDFs. Everything below uses train OOF probabilities (tuned LightGBM) only; student_test was not touched.
- `python scripts/hmm_error_analysis.py` -> `hmm_error_breakdown.csv`. Of 185,107 labelled train points, classifier+HMM gets 93.88% right. Wrong points: reference arc not in the candidates 0.92%, reverse-direction twin 1.92%, adjacent arc (junction / arc boundary) 2.18%, other 1.11%. Twin errors happen mostly at low speed (median 3.6-4.4 m/s on trips 63 and 75), where the GPS heading is noise.
- Tried: direction-aware same-arc transition (moving backwards along a directed arc costs a loop round the network). Train Top-1 dropped to 0.8845 (10 m slack) / 0.9318 (30 m) / 0.9319 (no backward cost). Reason found: 15% of consecutive point pairs on the *reference* arc move backwards by more than 2 m (first 40 train trips), so GPS noise makes "direction" unreliable. Rejected; code unchanged.
- `python scripts/hmm_mixed_emission.py` -> `hmm_mixed_emission_grid.csv` (160 settings): emission = w_clf*log p + w_dist*Gaussian(d). Best 0.9403 (w_clf 0.1, w_dist 0.5, beta 10, floor 25) vs 0.9388 now: +0.15 points, not adopted (too small to justify a change after the final run).
- Conclusion: the decoder is on a plateau around 0.94 on train. 95% point-level Top-1 is not reachable with these changes; the remaining errors are mostly junction boundaries and slow-speed twin roads, plus 0.9% candidate misses.

## E5: K=16 candidates + gap bridging + U-turn penalty (train only, adopted as v2)
Context: the team asked for at least 95% Top-1. Full experiment log with every setting tried: thread E's log (copied to `/mnt/project-files/handoff/work/E/EXPERIMENT_LOG.md`). student_test was not touched.
- Changes (commits 338a631, 0f43a82): K_CANDIDATES 8 -> 16 (`candidates.py`); the HMM now bridges time-gap segments instead of restarting Viterbi with no memory; switching to the reverse-direction twin arc costs a fixed `uturn` = 100 nats.
- `scripts/hmm_eval.py` grid widened to beta {5,10,15,20,30,50}, floor {25,50}, emission weight {0.03,0.05,0.07,0.1,0.25,0.5,1.0}.
- Command: `python scripts/hmm_eval.py --oof results/models/oof_lightgbm_tuned.parquet` (OOF from K=16 rebuild, `bayes_opt.py --reuse-best`).
- Headline (pooled Top-1 / route continuity, 185,107 labelled points, 80 trips): distance only 0.5536 / 0.8243; distance + HMM 0.9422 / 0.9802; classifier only 0.8504 / 0.9444; **classifier + HMM 0.9510 / 0.9806** (beta 10, floor 50, w 0.1, uturn 100).
- Honest caveat: the HMM parameters are chosen on the same OOF points they are scored on. Thread E's held-out check (choose params on half the trips, score the other half) gives 0.9480 / 0.9505, so v2 is at the 95% line, not clearly above it.
- Then `python scripts/export_model.py` rewrote `artifacts/` with these params. API smoke test (in-sample train trips): trip 1 0.998, trip 13 0.993, trip 90 0.927.
- The v1 (K=8) CSVs are kept in `results/pipeline/v1_k8/` and `results/models/v1_k8/`. The student_test numbers in `hmm_comparison_student_test.csv` are still the v1 one-time run.
