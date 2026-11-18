# Role D — API, UI & Deployment (Mahit)

Read [WORK_SPLIT.md](WORK_SPLIT.md) first for the project overview, data format and partition rules.

## Your job in one line

Two-laptop demo: a FastAPI backend on **laptop 1** runs the full prediction pipeline for a trip; a UI on **laptop 2** lets the evaluator pick one of 10 trips and shows the predicted route + accuracy.

## How the final evaluation runs (from the guidelines)

1. API running on laptop 1, UI on laptop 2.
2. UI lists the 10 reserved evaluation trips; evaluator picks one.
3. UI sends the **trip ID** to the API.
4. API runs candidates → classifier → HMM/Viterbi, computes accuracy against its **server-side** reference route.
5. UI displays predicted route + results.
6. Evaluator looks at the **API logs in the terminal of laptop 1** — so log every request clearly.

Must work for **all 10** trips. Hardcoded outputs or bypassing the API = disqualification territory.

We don't have the evaluator trips yet (asking organisers). Build against the 10 `student_test` trips and make the trip folder a setting (env var or CLI arg) so swapping it is one change.

## API (`api/`)

- FastAPI + uvicorn, run with `uvicorn api.main:app --host 0.0.0.0 --port 8000` so laptop 2 can reach it over LAN. On Windows allow the port through the firewall.
- Load model + configs **once at startup** (`models/classifier.joblib`, `models/classifier_config.json`, `models/hmm_config.json`).
- Endpoints (proposed):
  - `GET /trips` → list of trip IDs available
  - `POST /predict` body `{"trip_id": "..."}` → `{trip_id, gps_points: [[lat, lon], …], predicted_route: [[lat, lon], …] (polyline of predicted arcs), predicted_arcs: [...], metrics: {top1_accuracy, route_continuity, …}, inference_ms}`
- **Never** return the reference route or per-point labels to the UI. Only the computed metrics.
- Log line per request: time, client IP, trip_id, accuracy, latency.
- Enable CORS if the UI is a browser page served from another origin.

Until B/C deliver, use a **dummy predictor** (e.g. nearest arc per GPS point) behind the same function signature, so swapping in the real pipeline is one line. Clearly label it as dummy in code — it must not ship.

## UI (`ui/`)

Pick whatever you can explain and build fastest, e.g.:
- **Streamlit** + `streamlit-folium` (all Python), or
- a single HTML page with Leaflet.js calling the API via `fetch`.

Must show: the 10 trip choices, a map with raw GPS points and the predicted route (distinct colours, legend), and the metrics returned by the API. API URL must be configurable (laptop 1's LAN IP).

## Deployment & docs

- `requirements.txt` (pinned versions) at repo root.
- README sections: install, run API, run UI, how to point UI at the API, how the pipeline works.
- Test with two real laptops on the same Wi-Fi/hotspot before demo day; campus Wi-Fi may block device-to-device traffic — have a phone hotspot as backup.
- Screenshots for the report: working UI + successful API responses in laptop 1's terminal.

## Report section you draft

§7 (API, UI, deployment): architecture, endpoints, screenshots. Source code must match what's demoed.

## Viva prep

Be ready to explain: request flow end-to-end, how ground truth is kept server-side, how the model is loaded, and how you'd debug the UI not reaching the API.
