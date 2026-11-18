"""PathSnap API (laptop 1): runs the full map-matching pipeline for one trip per request.

Run from the repo root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Settings (environment variables):
    PATHSNAP_DATA_ROOT  folder holding the partition folders (default: "Student folder/Student folder")
    PATHSNAP_SPLIT      partition served to the UI (default: student_test; set to the evaluator folder name on demo day)

The reference route is read only after prediction, to score it; it is never returned to the UI.
"""
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import data, network, viterbi
from src.build_candidates import route_labels, trip_features

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("PATHSNAP_DATA_ROOT"):
    data.DATA_ROOT = Path(os.environ["PATHSNAP_DATA_ROOT"]).resolve()
SPLIT = os.environ.get("PATHSNAP_SPLIT", "student_test")
PARAMS = viterbi.HMMParams.from_json(ROOT / "models" / "hmm_config.json")
TRIPS = data.trip_ids(SPLIT)

app = FastAPI(title="PathSnap")
# the UI runs on another laptop (or from a local file), so allow cross-origin calls
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/trips")
def trips():
    return {"split": SPLIT, "trips": TRIPS}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    if req.trip_id not in TRIPS:            # also blocks any path tricks in trip_id
        raise HTTPException(404, f"unknown trip {req.trip_id!r}")
    t0 = time.perf_counter()

    trip = data.load_trip(SPLIT, req.trip_id)
    route = trip.pop("route")               # held back: prediction never sees it
    feats, xy, a, b = trip_features(trip)
    graph = network.RoadGraph(network.load_trip(req.trip_id, SPLIT, with_route=False))
    pred = viterbi.decode(feats, None, graph, PARAMS)   # None = distance-only emissions for now
    elapsed_ms = (time.perf_counter() - t0) * 1000

    metrics = {"n_points": len(pred), "route_continuity": network.route_continuity(pred, graph)}
    if route is not None:                   # score server-side; only the numbers leave the API
        r = route.arc_id.to_numpy(dtype=np.int64)
        pos, _ = route_labels(xy, a, b, r)
        metrics["top1_accuracy"] = float(np.mean(pred == r[pos]))

    nodes, arcs = trip["nodes"], trip["arcs"].to_numpy(dtype=np.int64)
    path = [int(x) for i, x in enumerate(pred) if x >= 0 and (i == 0 or x != pred[i - 1])]
    segments = [[[nodes.lat[u], nodes.lon[u]], [nodes.lat[v], nodes.lon[v]]] for u, v in arcs[path]]

    acc = metrics.get("top1_accuracy")
    print(f"{datetime.now():%H:%M:%S} {request.client.host if request.client else '?'} POST /predict "
          f"trip={req.trip_id} points={len(pred)} accuracy={'n/a' if acc is None else f'{acc:.4f}'} "
          f"time={elapsed_ms:.0f}ms", flush=True)

    return {
        "trip_id": req.trip_id,
        "gps_points": trip["track"][["lat", "lon"]].to_numpy().tolist(),
        "predicted_arcs": pred.tolist(),
        "predicted_route": segments,
        "metrics": metrics,
        "inference_ms": round(elapsed_ms, 1),
    }
