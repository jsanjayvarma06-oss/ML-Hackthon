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
|    ├── GET /models  ─── Available Live Classifiers          |
|    ├── GET /comparison ─ Benchmark Comparison Tables        |
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
|    └── artifacts/model.joblib & artifacts/models/*.joblib   |
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
|    ├── Model Selection Dropdown (from GET /models)          |
|    ├── Trip Selection Dropdown (from GET /trips)            |
|    ├── Reconstruct Route Trigger (POST /predict)            |
|    ├── Metric Dashboard (Top-1, Continuity, Timing)         |
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
  "status": "ok",
  "api_version": "1.0.0",
  "eval_dir": "student_test",
  "model": "lightgbm_tuned",
  "model_file_found": true
}
```

### 2.2 Available Models: `GET /models`
Returns list of models available to run for live comparisons:
- **Request**: `GET http://<SERVER_IP>:8000/models`
- **Response**:
```json
{
  "models": [
    "lightgbm_tuned",
    "knn_distance_weighted",
    "lightgbm",
    "logistic_regression",
    "mlp",
    "naive_bayes",
    "random_forest",
    "xgboost"
  ],
  "default": "lightgbm_tuned"
}
```

### 2.3 List Trips: `GET /trips`
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

### 2.4 Route Prediction: `POST /predict`
Performs candidate generation, ML inference, and HMM Viterbi decoding. Evaluates accuracy server-side and returns the predicted route.
- **Request**: `POST http://<SERVER_IP>:8000/predict`
```json
{
  "trip_id": "00000000",
  "model": "lightgbm_tuned"
}
```
- **Response**:
```json
{
  "success": true,
  "trip_id": "00000000",
  "model": "lightgbm_tuned",
  "n_points": 2503,
  "has_reference_route": true,
  "gps": [[12.971598, 77.594562], ...],
  "predicted_path": [[[12.9716, 77.5945], [12.9720, 77.5950]], ...],
  "predicted_route": [12, 12, 14, 15, ...],
  "metrics": {
    "top1_accuracy": 0.9512,
    "top1_accuracy_classifier_only": 0.9231,
    "route_continuity": 0.9984,
    "n_scored_points": 2503
  },
  "timing_ms": {
    "clean": 5.12,
    "candidates": 240.35,
    "features": 312.40,
    "hmm": 98.15,
    "total": 656.02
  },
  "processing_time_seconds": 0.656,
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
In the Streamlit sidebar:
- API URL: `http://<LAPTOP_1_IP>:8000`
- Model: Select from dropdown
- Trip: Select from dropdown
- Click **Reconstruct route**.

---

## 6. Security and Evaluation Integrity

1. **Reference Route Privacy**:
   - `.route` reference files are stored exclusively on Laptop 1.
   - The API evaluates predictions server-side and outputs only statistical metrics (`top1_accuracy`, `continuity`).
   - Reference arc sequences are strictly stripped from all outgoing responses.
2. **Directory Traversal Protection**:
   - `trip_id` inputs are strictly validated via regex `^[a-zA-Z0-9_-]{1,64}$` to prevent path traversal attempts (`../`).
3. **No Dynamic Training on Inference**:
   - The models are pre-loaded at server startup into memory. Inference does not retrain models.
