# PathSnap: API, UI and two-laptop deployment (Role D)

The API on **laptop 1** runs the full pipeline for one trip per request. The UI on **laptop 2**
lets the evaluator pick a trip and shows the raw GPS fixes, the predicted road route and the
accuracy computed on the server.

## Architecture

```
 Laptop 2 (browser)                         Laptop 1 (python -m api)
 ui/  React + Leaflet                       api/  FastAPI
 ┌──────────────────────┐   GET /trips      ┌───────────────────────────────────────────────┐
 │ trip list            │ ───────────────▶  │ resources.py  loaded once at startup:          │
 │ map: GPS + route     │   GET /trips/{id}/│   trip catalogue, hmm_config.json, classifier  │
 │ metrics, timings     │       track       │                                               │
 │                      │   POST /predict   │ pipeline.py   (never opens the .route file)   │
 │                      │ ───────────────▶  │   .track/.nodes/.arcs                          │
 │                      │                   │   → A: build_candidates.trip_features          │
 │                      │                   │   → B: classifier.predict_proba (if exported)  │
 │                      │                   │   → C: network.RoadGraph + viterbi.decode      │
 │                      │                   │   → lat/lon polylines of the predicted arcs    │
 │                      │ ◀───────────────  │ evaluation.py  reads .route only now; returns  │
 │                      │   JSON: points,   │   aggregate metrics, never the reference       │
 └──────────────────────┘   route, metrics  │ logs.py  one line per request in the terminal  │
                                            └───────────────────────────────────────────────┘
```

| File | Purpose |
|---|---|
| `api/config.py` | Settings from environment variables (no hard-coded paths or IPs) |
| `api/resources.py` | Startup loading: trip catalogue, role C's HMM params, role B's classifier; emission mode |
| `api/pipeline.py` | Inference: wires A → B → C on one trip, builds map geometry |
| `api/evaluation.py` | Server-side scoring after inference (only module that reads `.route`) |
| `api/schemas.py` | Request/response models (also the OpenAPI docs at `/docs`) |
| `api/logs.py` | Readable console log + rotating `logs/api.log` |
| `api/main.py` | FastAPI app, endpoints, error handling, CORS |
| `api/__main__.py` | `python -m api`: starts uvicorn on all interfaces and prints the LAN address |
| `tests/test_api.py` | Backend tests (22) |
| `ui/` | Vite + React + TypeScript + Tailwind + React-Leaflet frontend |

### How ground truth stays server-side

1. `pipeline.load_inputs` reads only `.track`, `.nodes`, `.arcs`; the network is built with `route=None`.
2. Candidates, classifier scores and Viterbi decoding finish first.
3. Only then does `evaluation.score` read `.route`, label each GPS point with role A's
   `route_labels` (the same labelling as `results/obs_labels_*.csv`) and count matches.
4. The response carries only aggregate numbers (accuracy, correct/total, arc overlap).
   No reference arcs, no per-point labels, no reference geometry.

Tests enforce this: the prediction is identical with the `.route` file deleted, the reference is
read after decoding, and `api/pipeline.py` contains no route access.

### Emission mode (classifier vs. distance baseline)

| `PATHSNAP_EMISSION` | Behaviour |
|---|---|
| `auto` (default) | Uses role B's `models/classifier.joblib` when present, otherwise role C's distance-only Gaussian emission |
| `classifier` | Requires the classifier; if it is missing, `/predict` answers 503 (no silent fallback) |
| `distance` | Always role C's distance baseline |

The mode in use is printed at startup, shown by `/health`, returned in every `/predict` response
and displayed in the UI. **Current state:** `models/classifier.joblib` has not been exported yet
(role B), so the demo runs `distance`. As soon as B commits `models/classifier.joblib` and
`models/classifier_config.json` (with `"feature_columns"` in training order, optional `"model"` name),
restart the API: it switches to classifier emissions automatically. Set
`PATHSNAP_EMISSION=classifier` for the final demo so a missing model can't go unnoticed.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `status`, `pipeline_ready`, `problems`, `emission`, classifier status, trip count, uptime |
| GET | `/trips` | Trips of the served partition with point count, duration, track length, bounding box |
| GET | `/trips/{trip_id}/track` | Raw GPS fixes `[[lat, lon], …]` for the map preview |
| POST | `/predict` | Body `{"trip_id": "00000040"}`; see below |
| GET | `/docs` | Interactive OpenAPI documentation |

`POST /predict` response:

```json
{
  "trip_id": "00000040", "split": "student_test",
  "gps_points": [[lat, lon], ...],
  "predicted_route": [[[lat, lon], ...], ...],
  "predicted_arcs": [arc_id per GPS point],
  "metrics": {"scored": true, "top1_accuracy": 0.922, "correct_points": 2761, "scored_points": 2995,
              "route_overlap": 0.925, "route_continuity": 0.996, "n_points": 2995},
  "pipeline": {"emission": "distance", "classifier": null, "n_candidates": 39712, ...},
  "timings_ms": {"load": 17, "candidates": 160, "classifier": 0, "decode": 230, "geometry": 3, "scoring": 40, "total": 450},
  "inference_ms": 410, "request_id": "a166aedb"
}
```

`predicted_route` is a list of connected polylines: a new piece starts wherever two consecutive
predicted arcs do not share a node, so gaps in the decoder output are shown, not papered over.

Errors always have the shape `{"detail", "code", "request_id", "stage"?}`:

| Status | `code` | When |
|---|---|---|
| 404 | `TRIP_NOT_FOUND` | Trip id not in the served partition |
| 422 | `INVALID_REQUEST` | Missing/empty/oversized trip id or characters outside `A-Z a-z 0-9 _ -` |
| 500 | `PIPELINE_ERROR` | A stage failed (`stage` = load, candidates, classifier, decoder); traceback in `logs/api.log` only |
| 503 | `PIPELINE_UNAVAILABLE` | Data, HMM config or required classifier missing (see `/health` problems) |

Metric definitions match the rest of the team: `top1_accuracy` = role C's `acc_viterbi`,
`route_overlap` = C's `arc_jaccard_viterbi`, `route_continuity` = `network.route_continuity`.
On the 10 student_test trips the API reproduces C's `results/viterbi_runs.csv` values exactly.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PATHSNAP_DATA_ROOT` | `Student folder/Student folder` | Folder containing the partition folders |
| `PATHSNAP_SPLIT` | `student_test` | Partition listed in the UI (set to the evaluator folder on demo day) |
| `PATHSNAP_MODELS_DIR` | `models` | Location of `hmm_config.json`, `classifier.joblib`, `classifier_config.json` |
| `PATHSNAP_EMISSION` | `auto` | `auto`, `classifier` or `distance` |
| `PATHSNAP_REQUIRE_CLASSIFIER` | `0` | `1` behaves like `PATHSNAP_EMISSION=classifier` |
| `PATHSNAP_CORS_ORIGINS` | `*` | Comma-separated UI origins allowed to call the API |
| `PATHSNAP_LOG_DIR` | `logs` | Where `api.log` is written |
| `PATHSNAP_PORT` | `8000` | Port for `python -m api` |
| `VITE_API_BASE_URL` (UI) | `http://<UI host>:8000` | API address baked into the UI; can be changed in the UI |

Relative paths are resolved from the repository root.

## Setup

### Laptop 1: API (Python 3.10–3.12)

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt # only to run the tests
```

### Laptop 2: UI (Node.js 20 or newer)

```bash
cd ui
npm ci
```

## Running on one laptop

```bash
python -m api                       # terminal 1, repository root
```

```bash
cd ui && npm run dev                # terminal 2
```

Open http://localhost:5173.

## Two-laptop demonstration

1. **Network.** Join both laptops to the same Wi-Fi or, more reliably, the same phone hotspot.
   Campus Wi-Fi often isolates devices from each other.
2. **Laptop 1: start the API** from the repository root:

   ```bash
   python -m api
   ```

   It prints `network: http://<IP>:8000`. That is the address for laptop 2. To look it up by hand:
   macOS `ipconfig getifaddr en0`, Windows `ipconfig` (IPv4 Address of the Wi-Fi adapter),
   Linux `hostname -I`.
   For the evaluator trips, point the API at their folder before starting, e.g. if they are in
   `Student folder/Student folder/evaluator/`:

   ```bash
   PATHSNAP_SPLIT=evaluator PATHSNAP_EMISSION=classifier python -m api
   ```

   Windows PowerShell: `$env:PATHSNAP_SPLIT="evaluator"; python -m api`.
3. **Laptop 1: firewall.** The first start on macOS asks to allow incoming connections for Python:
   allow it. On Windows run once in an administrator terminal:

   ```bash
   netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
   ```

4. **Laptop 2: check the connection** by opening `http://<IP>:8000/health` in the browser. It must
   show `"status":"ok"`.
5. **Laptop 2: start the UI** with the API address:

   ```bash
   cd ui
   echo VITE_API_BASE_URL=http://<IP>:8000 > .env.local
   npm run dev
   ```

   Open http://localhost:5173. The address can also be typed in the UI (**Change** next to the
   connection status); the browser remembers it.
6. **No Node on laptop 2?** Build once on any machine with the address set
   (`VITE_API_BASE_URL=http://<IP>:8000 npm run build`), copy the `ui/dist/` folder to laptop 2 and, from the
   folder that contains `dist`, run `python -m http.server 5173 -d dist`; open http://localhost:5173.

### Troubleshooting

| Symptom | Check |
|---|---|
| UI says "API not reachable" | Open `http://<IP>:8000/health` on laptop 2. If that fails too it is the network, not the UI |
| `/health` fails from laptop 2 but works on laptop 1 | API must listen on `0.0.0.0` (`python -m api` does); firewall rule; both on the same network; try the phone hotspot |
| IP changed | Hotspots hand out new addresses: re-read the `network:` line and update **Change** in the UI |
| `/health` shows `degraded` | Read `problems`: wrong `PATHSNAP_DATA_ROOT`/`PATHSNAP_SPLIT`, missing `models/hmm_config.json`, or classifier required but absent |
| Trip list empty | The partition folder has no `<id>/<id>.track/.nodes/.arcs` triples |
| Map tiles missing | Tiles come from Esri over the internet; the route and GPS points still draw without them |
| Prediction error with a `stage` | Look at the API terminal / `logs/api.log` on laptop 1 for the request id shown in the UI |

## Demo script

1. Laptop 1: `python -m api` (with `PATHSNAP_SPLIT` set to the evaluator folder). Point at the
   startup lines: trips found, HMM parameters, emission mode, `Ready`.
2. Laptop 2: open the UI; the dot next to "API connected" is green and the evaluator's trips are listed.
3. Click a trip: its raw GPS fixes appear (cyan, start ring and end square).
4. Click **Match route**: the API runs candidates → classifier → Viterbi; the predicted route draws
   over the fixes (amber), accuracy and timings appear on the left.
5. Show laptop 1's terminal: one line per request with client IP, trip, accuracy, continuity, time.
6. Zoom into a junction, toggle **Raw GPS fixes** / **Predicted route** in the legend, switch to
   **Satellite** to show the route sits on the road.
7. Repeat for further trips; each result stays in the list for comparison.

## Tests

```bash
python -m pytest tests -q           # backend: 22 tests
cd ui && npm run lint && npm run typecheck && npm test && npm run build
```
