# Evidence pack for the report

Everything here is built from files that already exist in `results/`. No model was retrained and
neither `student_test` nor `evaluator_test` was re-run to make this pack. Tables are rendered
images of the CSV values (rounded to 4 decimals); the full-precision CSVs are in `csv/`.
Map screenshots are headless-Chrome captures of the saved Folium HTML in `results/maps/`.

## Caveats (read before quoting any number)

- **Trip 5 (about 97.3%) is an in-sample TRAIN trip. It is not test accuracy.** All four route maps in
  item 08 (trips 5, 39, 46, 63) are train trips, so their predictions are in-sample. The file names
  say `TRAIN_INSAMPLE`.
- Cross-validation and out-of-fold (OOF) numbers (items 03, 04, 05, 06) use the 80 train trips with
  GroupKFold by trip. HMM parameters were fitted on the same train OOF, so those HMM numbers are
  mildly optimistic.
- `student_test` (item 07) is a **single run** (LightGBM tuned + HMM). `final_student_test.csv` has only
  6 models: XGBoost and Naive Bayes were not run on `student_test` (team decision).
- Evaluator trips were scored only after the system was frozen, and no result from them was used to change anything. Their logs are
  item 10, not in this folder's generated files.
- Item 01 `01_eda_gps_summary.png` is aggregated from `trip_inventory.csv` by the script that built
  this pack; its numbers match `results/eda/EXPERIMENTS.md`.
- The Folium road-network layer is a toggle in the saved HTML. In the zoomed-out EDA map screenshots
  the grey network is hard to see; the route and GPS points are visible.

## Files

| Item | File(s) | Numbers come from |
|---|---|---|
| 01 EDA | `01_eda_missing_values.png`, `01_eda_gps_summary.png`, `01_eda_sampling_interval_hist.png`, `01_eda_network_size.png`, `01_eda_trajectories.png` | `results/eda/missing_values.csv`, `trip_inventory.csv`, plots in `results/eda/`; also `csv/dataset_summary.csv`, `trip_stats.csv`, `candidate_miss_rate.csv` |
| 02 Folium road network + GPS | `02_folium_road_network_gps_trip_00000005.png`, `..._00000013.png`, `..._00000078.png` | `results/maps/eda_*.html` (train trips) |
| 03 Model comparison (7 models) | `03_model_comparison_table.png`, `03_model_comparison_plot_cv_top1.png`, `..._cv_f1_macro.png`, `..._cv_log_loss.png` | `csv/model_comparison.csv` |
| 04 GroupKFold folds | `04_groupkfold_val_top1_by_fold.png` | `csv/cv_folds.csv`, `csv/cv_folds_tuned.csv` |
| 05 Bayesian optimisation (Optuna) | `05_optuna_history.png`, `05_optuna_param_importance.png`, `05_optuna_top10_trials.png`, `05_optuna_best_params.png`, `05_tuned_vs_untuned.png` | `csv/optuna_trials.csv` (all trials), `csv/best_params.json`, `csv/tuned_vs_untuned.csv` |
| 06 Classifier-only vs HMM + Viterbi | `06_classifier_only_vs_hmm_viterbi.png` | `csv/hmm_comparison.csv` (LightGBM tuned), `csv/hmm_comparison_xgboost.csv`, `csv/hmm_comparison_naive_bayes.csv`; also `csv/hmm_grid.csv`, `hmm_params.json`, `hmm_error_breakdown.csv` |
| 07 student_test | `07_student_test_hmm.png`, `07_student_test_classifiers.png`, `07_student_test_top1_f1.png`, `07_student_test_log_loss.png` | `csv/hmm_comparison_student_test.csv`, `csv/final_student_test.csv` |
| 08 Predicted routes (train, in-sample) | `08_route_TRAIN_INSAMPLE_trip_00000005.png`, `..._00000039.png`, `..._00000046.png`, `..._00000063.png` | `results/maps/trip_*.html` |
| 09 UI screenshot | `09_ui_*.png` (to be added by Wolf) | live UI |
| 10 API logs | `10_api_logs.png` (to be added by Wolf) | Laptop 1 terminal |

Map legend (08): grey = road network, green = reference route, red = predicted route, blue = GPS points.
The reference route in these report maps is allowed because these are offline report figures; the live UI never shows it.
