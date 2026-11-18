"""Inference for one trip: GPS track + road network -> predicted arc per GPS point.

Every modelling step is the team's own code; this module only wires them together:

    load inputs (.track, .nodes, .arcs)        -- this module (the .route file is NOT read)
    candidates + features                      -- role A  src/build_candidates.trip_features
    emission scores                            -- role B  models/classifier.joblib (or C's distance baseline)
    HMM / Viterbi decoding                     -- role C  src/viterbi.decode on src/network.RoadGraph
    map geometry (lat/lon polylines)           -- this module

Ground truth is scored later and separately in ``api/evaluation.py``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import data, network, viterbi
from src.build_candidates import trip_features

from .resources import Resources


class PipelineError(Exception):
    """A stage of the pipeline failed. ``stage`` is safe to show to the UI; the cause is logged only."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.message = message


@dataclass
class Prediction:
    trip_id: str
    track: pd.DataFrame                  # lon, lat, t
    pred_arcs: np.ndarray                # one arc id per GPS point (-1 = no prediction)
    polylines: list[list[list[float]]]   # [[ [lat, lon], ... ], ...] connected pieces of the route
    xy: np.ndarray                       # projected GPS points (m), reused by scoring
    a: np.ndarray                        # projected arc start points (m)
    b: np.ndarray                        # projected arc end points (m)
    graph: network.RoadGraph
    stats: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)


def load_inputs(res: Resources, trip_id: str) -> dict:
    """The legitimate model inputs only: track and road network. Never the .route file."""
    base = res.trip_dir(trip_id) / trip_id
    # same parser/format as role A's data.load_trip, minus the reference route
    return {
        "track": data._read(base.with_suffix(".track"), ["lon", "lat", "t"]),
        "nodes": data._read(base.with_suffix(".nodes"), ["lon", "lat"]),
        "arcs": data._read(base.with_suffix(".arcs"), ["u", "v"], dtype="Int64"),
    }


def run(res: Resources, trip_id: str) -> Prediction:
    timings: dict[str, float] = {}

    def lap(name: str, t0: float) -> float:
        now = time.perf_counter()
        timings[name] = round((now - t0) * 1000, 1)
        return now

    t = time.perf_counter()
    try:
        inputs = load_inputs(res, trip_id)
        _check_inputs(inputs)
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError("load", "trip files could not be read") from exc
    t = lap("load", t)

    try:
        feats, xy, a, b = trip_features(inputs)                       # role A
    except Exception as exc:
        raise PipelineError("candidates", "candidate generation failed") from exc
    t = lap("candidates", t)

    emission = None                                                    # None = C's distance baseline
    if res.emission_mode == "classifier":
        if res.classifier is None:
            raise PipelineError("classifier", "classifier is not loaded")
        try:
            if len(feats):
                prob = res.classifier.positive_proba(feats)            # role B
                emission = (res.hmm_params.emission_weight              # tuned weight vs transitions
                            * viterbi.classifier_emission_logp(prob))  # role C's log transform
            else:
                emission = np.zeros(0)
        except Exception as exc:
            raise PipelineError("classifier", "classifier scoring failed") from exc
    t = lap("classifier", t)

    try:
        trip = network.Trip(                                           # role C, built without the route
            trip_id, res.settings.split,
            inputs["track"][["lon", "lat", "t"]].to_numpy(float),
            inputs["nodes"][["lon", "lat"]].to_numpy(float),
            inputs["arcs"].to_numpy(np.int64),
            route=None,
        )
        graph = network.RoadGraph(trip)
        pred, info = viterbi.decode(feats, emission, graph, res.hmm_params, return_info=True)
    except Exception as exc:
        raise PipelineError("decoder", "HMM/Viterbi decoding failed") from exc
    t = lap("decode", t)

    polylines, n_route_arcs, length_m, breaks = route_geometry(pred, inputs, graph)
    lap("geometry", t)

    stats = {
        "emission": res.emission_mode,
        "classifier": res.classifier.name if (res.classifier and res.emission_mode == "classifier") else None,
        "n_candidates": int(len(feats)),
        "points_without_candidates": int((info.n_candidates == 0).sum()),
        "segments": int(info.segment.max() + 1) if len(info) and info.segment.max() >= 0 else 0,
        "forced_breaks": int(info.forced_break.sum()),
        "n_route_arcs": n_route_arcs,
        "route_length_km": round(length_m / 1000, 3),
        "route_breaks": breaks,
        "route_continuity": _nan_to_none(network.route_continuity(pred, graph)),
    }
    return Prediction(trip_id, inputs["track"], pred, polylines, xy, a, b, graph, stats, timings)


def route_geometry(pred: np.ndarray, inputs: dict, graph: network.RoadGraph):
    """Predicted arcs -> connected lat/lon polylines.

    Consecutive duplicate arcs collapse to one. A new polyline starts wherever two consecutive
    predicted arcs do not share a node (e.g. a skipped short arc or a decoder segment break):
    gaps are shown honestly instead of drawing a straight line the decoder never chose.
    """
    seq = [int(x) for x in pred if x >= 0]
    path = [x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1]]
    if not path:
        return [], 0, 0.0, 0
    lat = inputs["nodes"].lat.to_numpy()
    lon = inputs["nodes"].lon.to_numpy()
    frm, to = graph.arc_from, graph.arc_to

    polylines, cur = [], []
    for k, arc in enumerate(path):
        u, v = int(frm[arc]), int(to[arc])
        if not cur or int(to[path[k - 1]]) != u:
            if cur:
                polylines.append(cur)
            cur = [[round(float(lat[u]), 7), round(float(lon[u]), 7)]]
        cur.append([round(float(lat[v]), 7), round(float(lon[v]), 7)])
    polylines.append(cur)
    length_m = float(graph.arc_len[np.unique(path)].sum())
    return polylines, len(set(path)), length_m, len(polylines) - 1


def _check_inputs(inputs: dict):
    track = inputs["track"]
    if len(track) == 0:
        raise PipelineError("load", "trip has no GPS points")
    if track[["lon", "lat", "t"]].isna().any().any():
        raise PipelineError("load", "trip has malformed GPS rows")
    if len(inputs["arcs"]) == 0 or len(inputs["nodes"]) == 0:
        raise PipelineError("load", "trip has an empty road network")
    arcs = inputs["arcs"]
    if arcs.isna().any().any() or int(arcs.max().max()) >= len(inputs["nodes"]) or int(arcs.min().min()) < 0:
        raise PipelineError("load", "road network references unknown nodes")


def _nan_to_none(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)
