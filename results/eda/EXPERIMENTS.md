# EDA experiment log (Samarth's files)

One entry per run. Numbers are copied from the output files named in the entry.

## Run 1: dataset inventory on train
- Command: `python scripts/eda.py` (about 40 s)
- Base commit: 1fbe5e7 (main 01ebede + features patch), samarth/eda-api-readme
- Partition: train only, 80 trips. student_test and evaluator not read.
- Outputs: `results/eda/trip_inventory.csv`, `missing_values.csv`, `sampling_hist.png`,
  `network_size.png`, `trajectories.png`; maps `results/maps/eda_00000005.html`,
  `eda_00000013.html`, `eda_00000078.html`
- Headline numbers (from trip_inventory.csv):
  - 208,417 GPS points; median 1,529 per trip (361 to 19,313)
  - arcs per trip 649 to 307,392; median 20 weakly connected components, 1 trip with a single component
  - median time step 1 s everywhere; fractional timestamps in 3 trips (2, 71, 97)
  - 22 trips have a step over 60 s; largest 153,064 s (trip 90), 52,810 s (trip 78)
  - 10 trips have speeds over 70 m/s (max 180.4 m/s); 53 trips have duplicate arcs; 8 have self-loops
  - 10 trips have unconnected steps in the reference route (max 3 per trip)
  - missing_values.csv: 0 non-finite rows, 0 non-increasing time steps, 0 out-of-range route ids
- Definitions: gap = step > 60 s; spike = implied speed > 70 m/s between consecutive points with dt > 0;
  components = weakly connected components of the arc graph.
