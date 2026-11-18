"""TEMPORARY stand-ins for role A's outputs, so the HMM can be built before A's files land.

* ``simple_candidates`` -> same schema as ``results/candidates_*.parquet``
  (trip_id, obs_idx, arc_id, label, f_dist_m) using a plain radius + top-K search.
* ``reference_labels``  -> same idea as ``results/obs_labels_*.csv``
  (one reference arc per GPS point).

Once A pushes the real files, the decoder should read those instead
(see ``load_candidates`` / ``load_obs_labels`` in run_viterbi.py). Nothing here is
used inside ``viterbi.decode`` except the candidate list itself; the reference route
is used **only** to build labels for scoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .network import RoadGraph, Trip, point_to_segments


def simple_candidates(trip: Trip, graph: RoadGraph, ref_arcs: np.ndarray | None = None,
                      radius_m: float = 50.0, k: int = 8, densify_m: float = 10.0) -> pd.DataFrame:
    """For each GPS point: the ``k`` nearest arcs within ``radius_m`` (exact point-to-segment distance).

    Arcs are densified to points every ``densify_m`` metres and put in a KD-tree for the
    coarse search, then exact distances are computed for the hits.
    """
    # densify arcs -> sample points tagged with their arc id
    n_pts = np.maximum(1, np.ceil(graph.arc_len / densify_m).astype(int)) + 1
    arc_ids = np.repeat(np.arange(len(n_pts)), n_pts)
    starts = np.cumsum(n_pts) - n_pts
    t = (np.arange(n_pts.sum()) - np.repeat(starts, n_pts)) / np.repeat(n_pts - 1, n_pts)
    sx = graph.ax[arc_ids] + t * (graph.bx[arc_ids] - graph.ax[arc_ids])
    sy = graph.ay[arc_ids] + t * (graph.by[arc_ids] - graph.ay[arc_ids])
    tree = cKDTree(np.column_stack([sx, sy]))

    gps = trip.gps_xy
    hits = tree.query_ball_point(gps, r=radius_m + densify_m / 2)
    rows = []
    for i, h in enumerate(hits):
        if not h:
            continue
        cand = np.unique(arc_ids[h])
        d, _ = point_to_segments(gps[i, 0], gps[i, 1], graph.ax[cand], graph.ay[cand],
                                 graph.bx[cand], graph.by[cand])
        keep = d <= radius_m
        cand, d = cand[keep], d[keep]
        order = np.argsort(d, kind="stable")[:k]
        for a, dd in zip(cand[order], d[order]):
            rows.append((i, int(a), float(dd)))
    df = pd.DataFrame(rows, columns=["obs_idx", "arc_id", "f_dist_m"])
    df.insert(0, "trip_id", trip.trip_id)
    if ref_arcs is not None:
        df.insert(3, "label", (df["arc_id"].to_numpy() == ref_arcs[df["obs_idx"].to_numpy()]).astype(int))
    else:
        df.insert(3, "label", -1)
    return df


def reference_labels(trip: Trip, graph: RoadGraph, max_dist_m: float = 100.0) -> np.ndarray:
    """Assign every GPS point one arc of the reference route (scoring only).

    ``.route`` lists arcs in driving order but has fewer rows than ``.track``, so we
    align them with a monotone dynamic programme: point t gets route position j(t),
    j is non-decreasing in t, and the total point-to-arc distance is minimised.
    Points farther than ``max_dist_m`` from their aligned arc get -1 (no reference).
    """
    route = trip.route
    gps = trip.gps_xy
    T, R = len(gps), len(route)
    D, _ = point_to_segments(gps[:, 0:1], gps[:, 1:2], graph.ax[route][None], graph.ay[route][None],
                             graph.bx[route][None], graph.by[route][None])  # (T, R)
    back = np.zeros((T, R), np.int32)
    cost = D[0].copy()
    for t in range(1, T):
        # prefix-min over j' <= j of previous cost
        idx = np.arange(R)
        best_idx = np.maximum.accumulate(np.where(cost == np.minimum.accumulate(cost), idx, 0))
        back[t] = best_idx
        cost = D[t] + cost[best_idx]
    j = np.empty(T, np.int64)
    j[-1] = int(np.argmin(cost))
    for t in range(T - 1, 0, -1):
        j[t - 1] = back[t, j[t]]
    ref = route[j].copy()
    ref[D[np.arange(T), j] > max_dist_m] = -1
    return ref
