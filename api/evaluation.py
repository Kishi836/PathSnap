"""Server-side scoring of a finished prediction against the reference route.

This is the ONLY API module that reads a ``.route`` file, and it is called only after
``pipeline.run`` has returned. Only aggregate numbers leave this module: no reference arcs,
no per-point labels, no reference geometry.

Accuracy uses exactly the labelling the rest of the team uses: role A's ``route_labels``
assigns each GPS point its reference arc (monotone along the route), the same definition
as ``results/obs_labels_*.csv``, role B's Top-1 accuracy and role C's ``acc_viterbi``.
"""
from __future__ import annotations

import numpy as np

from src.build_candidates import route_labels
from src.run_viterbi import jaccard

from .pipeline import Prediction
from .resources import Resources


def load_reference(res: Resources, trip_id: str) -> np.ndarray | None:
    path = res.trip_dir(trip_id) / f"{trip_id}.route"
    if not path.is_file():
        return None
    route = np.loadtxt(path, dtype=np.int64, ndmin=1)
    return route


def score(res: Resources, pred: Prediction) -> dict:
    """Aggregate metrics, or ``{"scored": False}`` when the partition has no reference route."""
    route = load_reference(res, pred.trip_id)
    if route is None or len(route) == 0:
        return {"scored": False}
    if route.min() < 0 or route.max() >= len(pred.a):
        raise ValueError("reference route references unknown arcs")
    pos, _ = route_labels(pred.xy, pred.a, pred.b, route)
    ref = route[pos]
    correct = int(np.sum(pred.pred_arcs == ref))
    total = int(len(ref))
    return {
        "scored": True,
        "top1_accuracy": correct / total if total else None,
        "correct_points": correct,
        "scored_points": total,
        "route_overlap": float(jaccard(pred.pred_arcs, route)),   # Jaccard of arc sets (C's arc_jaccard)
    }
