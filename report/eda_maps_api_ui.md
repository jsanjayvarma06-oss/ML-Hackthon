# Dataset exploration, maps, API and deployment

All numbers below come from files in `results/eda/`, produced by `python scripts/eda.py`
on the 80 **train** trips only. student_test and evaluator trips were not read.

## 1. Dataset exploration (Deliverables 2.2)

**Source files:** `results/eda/trip_inventory.csv` (one row per trip),
`results/eda/missing_values.csv`, figures `sampling_hist.png`, `network_size.png`,
`trajectories.png`.

Each trip has four files: `.track` (lon, lat, time), `.nodes`, `.arcs` (directed
node pairs) and `.route` (the reference route as an ordered list of arc ids, not one row
per GPS point). Node and arc ids are local to a trip.

### Size
- 80 trips, 208,417 GPS points in total; median 1,529 points per trip, range 361 to
  19,313.
- Networks range from 649 to 307,392 arcs (`network_size.png`). A spatial index is
  needed for candidate search on the largest ones.
- Each trip's network has a median of 20 weakly connected components (only 1 trip has a
  single component), so a route cannot be assumed to exist between any two arcs.

### Sampling
- The median time step is 1 s for every trip (`sampling_hist.png`). Three trips have
  fractional timestamps (2, 71, 97).
- 22 trips have at least one time step above 60 s. The largest are trip 90 (153,064 s,
  about 42.5 h) and trip 78 (52,810 s). A gap this large must split the trip into
  separate segments before matching.
- No time steps with dt <= 0 and no non-finite rows were found (`missing_values.csv`).

### Noise
- 10 trips contain implied speeds above 70 m/s between consecutive points (trips 1, 8,
  13, 32, 33, 43, 53, 57, 79, 92); the largest speed seen is 180 m/s. These are GPS jumps and
  should be removed in cleaning.

### Network quirks
- 53 trips contain duplicate arcs (same from/to pair twice); 8 contain self-loops
  (trips 13, 24, 35, 38, 55, 79, 84, 99). Candidate arcs must be de-duplicated.
- Reference routes in 10 trips contain a step where the next arc does not start where the
  previous one ends (trips 2, 19, 33, 41, 42, 69, 75, 80, 86, 89), at most 3 per trip.
  Labels from alignment must tolerate this.
- No route arc id was outside its arcs file; no arc referenced a missing node.

### What this means for the pipeline
Trips are worldwide, so distances are computed in metres per trip, never as Euclidean
distance on degrees. Timestamp gaps, speed spikes, duplicate arcs and disconnected
networks are handled in cleaning, candidate generation and the HMM transition model
rather than ignored.

## 2. Maps

`results/maps/eda_00000005.html`, `eda_00000013.html` (largest network) and
`eda_00000078.html` (52,810 s gap) were made with `pathsnap.viz.trip_map`. Layers: GPS
points (blue), nearby road network (grey), reference route (green). Raw files store lon,
lat; Folium needs lat, lon, and the swap happens in one place in `viz.py`.

## 3. API, UI and deployment (section 7)

- `api/main.py` (FastAPI) exposes `GET /health`, `GET /trips` and `POST /predict`.
  `/predict` runs the full pipeline live on the requested trip; nothing is cached or
  hard-coded. It reads the reference route only to compute accuracy, and returns GPS
  points, the predicted route and the metrics, never the reference route.
- `/health` reports the model name from the exported bundle so the UI shows which model
  ran. A missing `artifacts/model.joblib` gives a 503 with the fix, not a crash.
- The Streamlit UI (`ui/app.py`, Laptop 2) calls only the API.
- Deployment: Laptop 1 runs `uvicorn api.main:app --host 0.0.0.0 --port 8000` with
  `PATHSNAP_EVAL_DIR` pointing at the evaluator trips; Laptop 2 runs
  `streamlit run ui/app.py` and enters `http://<laptop-1-ip>:8000`. Both laptops share a
  network; the firewall must allow TCP 8000. Full steps are in `README.md`.
- Not yet measured: end-to-end API latency per trip. Add it from the API log lines
  once the pipeline is finished; nothing is claimed here.
