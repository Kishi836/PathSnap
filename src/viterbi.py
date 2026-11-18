"""HMM map-matching + Viterbi decoding (role C).

HMM definition (report section 5)
---------------------------------
* Time steps   : GPS points of one trip that have at least one candidate arc.
* Hidden state : at step t, which of point t's candidate arcs the car is on
                 (state j at step t  <->  candidate row j of obs_idx t -> arc_id).
* Emission     : log B_j(o_t)
                 - baseline : Gaussian on point-to-arc distance d,  -d^2 / (2 sigma^2)
                 - later    : log of the classifier probability for (point, arc)
* Transition   : Newson & Krumm (2009). With d_gc = great-circle distance between the
                 two GPS fixes and d_route = driving distance between the two snapped
                 positions on the network,
                     log A_ij = -|d_route - d_gc| / beta_eff,   beta_eff = beta * max(1, dt)
                 Unreachable within the search radius -> log_unreachable (tiny but > 0
                 probability, so decoding never dead-ends).
* Decoding     : delta_t(j) = max_i [delta_{t-1}(i) + log A_ij] + log B_j(o_t),
                 with back-pointers, all in log-space (products of hundreds of small
                 probabilities underflow float64; sums of logs do not).

Edge cases
----------
* Point with no candidates -> not an HMM step; after decoding it gets the previous
  decoded arc (or the next one at the start of a trip). Flagged as ``filled``.
* Time gap > gap_break_s   -> the trip is split into independent segments, each
  decoded on its own (the network path across a long gap is too uncertain).
* No connected path        -> transition gets log_unreachable; Viterbi still picks
  the best remaining option, and we count such "forced breaks" in the output.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .network import RoadGraph, haversine_m, point_to_segments


@dataclass
class HMMParams:
    sigma_m: float = 5.0           # GPS noise std (emission)
    beta_m: float = 3.0            # transition scale per 1 s step
    backward_tol_m: float = 10.0   # same-arc backward jitter treated as standing still
    gap_break_s: float = 60.0      # split trip into segments at larger time gaps
    log_unreachable: float = -1000.0
    search_factor: float = 2.0     # Dijkstra radius = factor * d_gc + slack
    search_slack_m: float = 300.0

    def to_json(self, path, **extra):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({**asdict(self), **extra}, indent=2))

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def distance_emission_logp(dist_m: np.ndarray, sigma_m: float) -> np.ndarray:
    """Baseline emission: log of a Gaussian on point-to-arc distance (constant dropped)."""
    return -0.5 * (np.asarray(dist_m) / sigma_m) ** 2


def classifier_emission_logp(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Emission from B's classifier: log P(candidate is the true arc)."""
    return np.log(np.clip(np.asarray(prob, float), eps, 1.0))


def _snap(graph: RoadGraph, gps_xy: np.ndarray, obs_idx: np.ndarray, arc_id: np.ndarray):
    """Exact distance and fraction along each candidate arc (recomputed, so we do not
    depend on which f_* columns A's table happens to contain)."""
    p = gps_xy[obs_idx]
    return point_to_segments(p[:, 0], p[:, 1], graph.ax[arc_id], graph.ay[arc_id],
                             graph.bx[arc_id], graph.by[arc_id])


def decode(cands: pd.DataFrame, emission_logp: np.ndarray | None, graph: RoadGraph,
           params: HMMParams | None = None, return_info: bool = False):
    """Viterbi decode one trip.

    cands         : rows (obs_idx, arc_id) for ONE trip (any extra columns ignored).
    emission_logp : log B per row of ``cands`` (same order). None -> distance-only baseline.
    Returns an int array, one arc_id per GPS point of the track (-1 if the trip had no
    candidates at all). With ``return_info`` also a per-point DataFrame of diagnostics.
    """
    p = params or HMMParams()
    trip = graph.trip
    T = len(trip.track)
    gps_xy = trip.gps_xy
    lon, lat, ts = trip.track[:, 0], trip.track[:, 1], trip.track[:, 2]

    obs = cands["obs_idx"].to_numpy(np.int64)
    arc = cands["arc_id"].to_numpy(np.int64)
    dist, frac = _snap(graph, gps_xy, obs, arc)
    em = distance_emission_logp(dist, p.sigma_m) if emission_logp is None else np.asarray(emission_logp, float)

    # group candidate rows by GPS point, keeping exact row <-> arc_id mapping
    order = np.argsort(obs, kind="stable")
    obs, arc, frac, em = obs[order], arc[order], frac[order], em[order]
    steps, starts = np.unique(obs, return_index=True)
    bounds = list(starts) + [len(obs)]
    S = [(int(steps[k]), arc[bounds[k]:bounds[k + 1]], frac[bounds[k]:bounds[k + 1]], em[bounds[k]:bounds[k + 1]])
         for k in range(len(steps))]

    pred = np.full(T, -1, np.int64)
    seg_id = np.full(T, -1, np.int64)
    forced = np.zeros(T, bool)  # transition into this point was "unreachable"

    # split into segments at large time gaps
    segments, cur = [], []
    for s in S:
        if cur and ts[s[0]] - ts[cur[-1][0]] > p.gap_break_s:
            segments.append(cur)
            cur = []
        cur.append(s)
    if cur:
        segments.append(cur)

    for sid, seg in enumerate(segments):
        delta = seg[0][3].copy()
        backptrs = []
        unreach = []
        for (t0, a0, f0, _), (t1, a1, f1, e1) in zip(seg[:-1], seg[1:]):
            d_gc = float(haversine_m(lon[t0], lat[t0], lon[t1], lat[t1]))
            dt = max(1.0, float(ts[t1] - ts[t0]))
            beta = p.beta_m * dt
            limit = p.search_factor * d_gc + p.search_slack_m
            logA = np.empty((len(a0), len(a1)))
            for i in range(len(a0)):
                for j in range(len(a1)):
                    rd = graph.route_distance(int(a0[i]), float(f0[i]), int(a1[j]), float(f1[j]),
                                              limit, p.backward_tol_m)
                    logA[i, j] = p.log_unreachable if not np.isfinite(rd) else -abs(rd - d_gc) / beta
            scores = delta[:, None] + logA
            bp = np.argmax(scores, axis=0)
            backptrs.append(bp)
            unreach.append(logA[bp, np.arange(len(a1))] <= p.log_unreachable)
            delta = scores[bp, np.arange(len(a1))] + e1

        # backtrack
        j = int(np.argmax(delta))
        for k in range(len(seg) - 1, -1, -1):
            t, a = seg[k][0], seg[k][1]
            pred[t] = a[j]
            seg_id[t] = sid
            if k > 0:
                forced[t] = unreach[k - 1][j]
                j = int(backptrs[k - 1][j])

    # points without candidates: carry previous state (or next at the start)
    filled = pred < 0
    if (~filled).any():
        idx = np.where(~filled, np.arange(T), -1)
        prev = np.maximum.accumulate(idx)
        first = int(np.argmax(~filled))
        prev[prev < 0] = first
        pred = pred[prev]
    if not return_info:
        return pred
    n_cand = np.bincount(obs, minlength=T)
    info = pd.DataFrame({"obs_idx": np.arange(T), "pred_arc": pred, "n_candidates": n_cand,
                         "segment": seg_id, "filled": filled, "forced_break": forced})
    return pred, info


def top1(cands: pd.DataFrame, score: np.ndarray, T: int) -> np.ndarray:
    """Independent per-point argmax ("classifier-only") prediction, -1 where no candidates."""
    df = pd.DataFrame({"obs_idx": cands["obs_idx"].to_numpy(), "arc_id": cands["arc_id"].to_numpy(),
                       "s": np.asarray(score, float)})
    best = df.loc[df.groupby("obs_idx")["s"].idxmax()]
    out = np.full(T, -1, np.int64)
    out[best["obs_idx"].to_numpy()] = best["arc_id"].to_numpy()
    return out
