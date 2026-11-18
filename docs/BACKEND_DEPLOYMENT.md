# PathSnap Backend Server & Two-Laptop Deployment Guide

This guide describes the architecture, startup, network configuration, and testing procedures for the **PathSnap** distributed map-matching and route reconstruction system operating across two separate Windows laptops.

---

## 1. Architecture Overview

```
+-------------------------------------------------------------+
|                     LAPTOP 1 (SERVER)                       |
|                                                             |
|  IP: 192.168.x.x (or LAN / Hotspot IP)                     |
|  Port: 8000 (0.0.0.0:8000)                                  |
|                                                             |
|  [FastAPI Backend (Uvicorn)]                                |
|    │                                                        |
|    ├── GET /health  ─── Health & Model Status               |
|    ├── GET /trips   ─── Authorised Trip IDs                 |
|    └── POST /predict ── End-to-End Reconstructor            |
|          │                                                  |
|          ├── 1. GPS Preprocessing & Cleaning (clean_track)  |
|          ├── 2. Spatial KD-Tree Candidates (generate_cands) |
|          ├── 3. Feature Extraction (extract_features)       |
|          ├── 4. ML Probability Classifier (model.joblib)    |
|          ├── 5. HMM + Viterbi Global Decoding               |
|          ├── 6. Server-Side Metric Scoring (Never Leaked)   |
|          └── 7. Lat/Lon Line Segments Reconstruction        |
|                                                             |
|  [Server-Side Data & Security Boundary]                     |
|    ├── data/raw/train/ & data/raw/student_test/             |
|    ├── Ground-truth reference routes (.route)               |
|    └── artifacts/model.joblib (Trained Estimator)           |
+-------------------------------------------------------------+
                              ▲
                              │ HTTP REST API (TCP)
                              │ JSON Request / Response
                              │ CORS Enabled (*)
                              ▼
+-------------------------------------------------------------+
|                     LAPTOP 2 (CLIENT)                       |
|                                                             |
|  [Streamlit UI / Frontend App]                              |
|    ├── Configurable API URL: http://<LAPTOP-1-IP>:8000      |
|    ├── Trip Selection Dropdown (from GET /trips)            |
|    ├── Reconstruct Route Trigger (POST /predict)            |
|    ├── Metric Dashboard (Top-1, Continuity, Macro F1, Time) |
|    └── Interactive Folium Map (GPS dots + Red Route)        |
+-------------------------------------------------------------+
```

---

## 2. API Endpoints Specification

### 2.1 Health Check: `GET /health`
Returns the status of the server, whether the ML model is loaded in memory, and dataset availability.
- **Request**: `GET http://<SERVER_IP>:8000/health`
- **Response**:
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "model_loaded": true,
  "model_name": "logistic_regression_scaled",
  "model_error": null,
  "data_available": true,
  "evaluation_trips_count": 10,
  "eval_dir": "student_test"
}
```

### 2.2 List Trips: `GET /trips`
Returns all authorised evaluation trip IDs available on the server. Never exposes labels, tracks, or route files.
- **Request**: `GET http://<SERVER_IP>:8000/trips`
- **Response**:
```json
{
  "trips": [
    "00000000",
    "00000007",
    "00000009",
    "00000040",
    "00000052",
    "00000056",
    "00000065",
    "00000076",
    "00000077",
    "00000091"
  ]
}
```

### 2.3 Route Prediction: `POST /predict`
Performs candidate generation, ML inference, and HMM Viterbi decoding. Evaluates accuracy server-side and returns the predicted route.
- **Request**: `POST http://<SERVER_IP>:8000/predict`
```json
{
  "trip_id": "00000000"
}
```
- **Response**:
```json
{
  "success": true,
  "trip_id": "00000000",
  "n_points": 2503,
  "gps": [[12.971598, 77.594562], ...],
  "predicted_path": [[[12.9716, 77.5945], [12.9720, 77.5950]], ...],
  "predicted_route": [12, 12, 14, 15, ...],
  "metrics": {
    "top1_accuracy": 0.9604,
    "top1_accuracy_classifier_only": 0.9305,
    "macro_f1": 0.8842,
    "log_loss": null,
    "route_continuity": 0.9988,
    "n_scored_points": 2503
  },
  "timing_ms": {
    "clean": 4.93,
    "candidates": 280.79,
    "features_and_ml": 398.3,
    "hmm_viterbi": 109.62,
    "total": 793.64
  },
  "processing_time_seconds": 0.7936,
  "message": "Prediction completed successfully"
}
```

---

## 3. Network Configuration for Two Windows Laptops

Both laptops must be connected to the **same network** (e.g., standard Wi-Fi router or Mobile Hotspot).

### Step 3.1: Find Laptop 1's Local IP Address
On **Laptop 1**, open PowerShell or CMD:
```powershell
ipconfig
```
Look for **IPv4 Address** under `Wireless LAN adapter Wi-Fi` (e.g. `192.168.1.15` or `192.168.43.100`).

### Step 3.2: Configure Windows Defender Firewall on Laptop 1
Allow inbound TCP traffic on port 8000:

**PowerShell (Run as Administrator):**
```powershell
New-NetFirewallRule -DisplayName "PathSnap Backend Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

Alternatively via Command Prompt (Administrator):
```cmd
netsh advfirewall firewall add rule name="PathSnap Backend Port 8000" dir=in action=allow protocol=TCP localport=8000
```

---

## 4. Server Startup Instructions (Laptop 1)

1. Navigate to the repository root directory:
```powershell
cd "c:\Users\Jagadeesh R\OneDrive\Desktop\S\ML\hackathon"
```

2. Set the evaluation data directory (point to `data/raw/student_test` or `data/evaluator` on demo day):
```powershell
$env:PATHSNAP_EVAL_DIR = "data/raw/student_test"
```

3. Start the FastAPI server on `0.0.0.0:8000`:
```powershell
& "C:\Users\Jagadeesh R\AppData\Local\Programs\Python\Python311\python.exe" -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

You will see the startup banner:
```
[SERVER] PathSnap backend starting...
[MODEL] Model loaded successfully: logistic_regression_scaled (trained on: train (15 trips))
[DATA] Evaluation dataset verified: 10 trips available in student_test
[NETWORK] Server listening on 0.0.0.0:8000 (accessible across LAN / Wi-Fi)
```

---

## 5. Client Connection Instructions (Laptop 2)

On **Laptop 2** (Friend's Laptop):

### Testing Connection from Laptop 2 Terminal
Using PowerShell:
```powershell
# 1. Health check
Invoke-RestMethod -Uri "http://<LAPTOP_1_IP>:8000/health" -Method Get

# 2. List available evaluation trips
Invoke-RestMethod -Uri "http://<LAPTOP_1_IP>:8000/trips" -Method Get

# 3. Test route reconstruction
Invoke-RestMethod -Uri "http://<LAPTOP_1_IP>:8000/predict" -Method Post -ContentType "application/json" -Body '{"trip_id":"00000000"}'
```

Using cURL (from CMD or Git Bash):
```bash
curl http://<LAPTOP_1_IP>:8000/health
curl http://<LAPTOP_1_IP>:8000/trips
curl -X POST http://<LAPTOP_1_IP>:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\":\"00000000\"}"
```

### Running Frontend Streamlit App on Laptop 2
```powershell
pip install -r ui/requirements.txt
streamlit run ui/app.py
```
In the Streamlit sidebar, enter:
`http://<LAPTOP_1_IP>:8000`
Select any trip ID (e.g. `00000000`) and click **Reconstruct route**.

---

## 6. Security and Evaluation Integrity

1. **Reference Route Privacy**:
   - `.route` reference files are stored exclusively on Laptop 1.
   - The API evaluates predictions server-side and outputs only statistical metrics (`top1_accuracy`, `continuity`, `macro_f1`).
   - Reference arc sequences are strictly stripped from all outgoing responses.
2. **Directory Traversal Protection**:
   - `trip_id` inputs are strictly validated via regex `^[a-zA-Z0-9_-]{1,64}$` to prevent path traversal attempts (`../`).
3. **No Dynamic Training on Inference**:
   - The model is loaded once at server startup into memory. Inference does not retrain models.

---

## 7. Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| Client receives `Connection Refused` or `Timeout` | Windows Firewall blocking port 8000 | Run the firewall rule command in Section 3.2 on Laptop 1 |
| Client cannot ping Laptop 1 | Laptops on different Wi-Fi networks | Connect both laptops to the same Wi-Fi router or phone mobile hotspot |
| `GET /trips` returns 500 | `PATHSNAP_EVAL_DIR` folder missing | Set `$env:PATHSNAP_EVAL_DIR = "data/raw/student_test"` before running uvicorn |
| `model_loaded: false` in `/health` | Missing `artifacts/model.joblib` | Model will fall back to Gaussian distance emission, or regenerate `artifacts/model.joblib` |
