"""Road-network helpers for PathSnap (role C, shared with B).

Loading, projection and point-to-segment geometry come from A's ``src/data.py``, so
the decoder sees exactly the same coordinates as the candidate features. This module
adds the graph questions the HMM needs:

* ``connected(a, b)``      -- same arc, or arcs[a].to == arcs[b].from
* ``route_distance(...)``  -- driving distance between two positions on arcs
                              (bounded Dijkstra on the node graph, weights in metres)
* ``route_continuity(...)``-- metric: share of consecutive predictions that are connected

Distances: arc lengths and GPS-to-GPS distances use the haversine (great-circle)
formula; point-to-arc distances use A's local equirectangular projection (metres).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import data

EARTH_R = data.EARTH_R
ROOT = Path(__file__).resolve().parents[1]


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def list_trips(split: str) -> list[str]:
    return data.trip_ids(split)


@dataclass
class Trip:
    """One trip as numpy arrays (built from A's ``data.load_trip``)."""
    trip_id: str
    split: str
    track: np.ndarray            # (T, 3) lon, lat, t
    nodes: np.ndarray            # (N, 2) lon, lat
    arcs: np.ndarray             # (A, 2) from_node, to_node
    route: np.ndarray | None     # (R,) reference arc ids (scoring only, never used in decode)
    proj: data.LocalProjection = field(init=False)

    def __post_init__(self):
        # same projection centre as A's trip_features (track mean)
        self.proj = data.LocalProjection(self.track[:, 1].mean(), self.track[:, 0].mean())

    @property
    def gps_xy(self):
        return self.proj(self.track[:, 0], self.track[:, 1])

    @property
    def node_xy(self):
        return self.proj(self.nodes[:, 0], self.nodes[:, 1])


def load_trip(trip_id: str, split: str = "train", with_route: bool = True) -> Trip:
    t = data.load_trip(split, trip_id)
    route = t["route"].arc_id.to_numpy(np.int64) if (with_route and t["route"] is not None) else None
    return Trip(trip_id, split, t["track"][["lon", "lat", "t"]].to_numpy(float),
                t["nodes"][["lon", "lat"]].to_numpy(float), t["arcs"].to_numpy(np.int64), route)


class RoadGraph:
    """Directed road graph of one trip. Arcs are straight segments between nodes."""

    def __init__(self, trip: Trip, dijkstra_cache_size: int = 50_000):
        self.trip = trip
        self.arc_from = trip.arcs[:, 0]
        self.arc_to = trip.arcs[:, 1]
        n = len(trip.nodes)
        lon, lat = trip.nodes[:, 0], trip.nodes[:, 1]
        self.arc_len = haversine_m(lon[self.arc_from], lat[self.arc_from], lon[self.arc_to], lat[self.arc_to])
        self.node_xy = trip.node_xy
        self.a_xy = self.node_xy[self.arc_from]   # (A, 2) arc start, metres
        self.b_xy = self.node_xy[self.arc_to]     # (A, 2) arc end, metres

        # CSR adjacency (node -> node, weight = arc length). Keep the shortest of parallel arcs.
        order = np.lexsort((self.arc_len, self.arc_to, self.arc_from))
        f, t, w = self.arc_from[order], self.arc_to[order], self.arc_len[order]
        keep = np.ones(len(f), bool)
        keep[1:] = (f[1:] != f[:-1]) | (t[1:] != t[:-1])
        f, t, w = f[keep], t[keep], w[keep]
        self.indptr = np.zeros(n + 1, np.int64)
        np.add.at(self.indptr, f + 1, 1)
        self.indptr = np.cumsum(self.indptr)
        self.nbr = t
        self.nbr_w = w

        self._cache: dict[int, tuple[float, dict[int, float]]] = {}
        self._cache_size = dijkstra_cache_size

    def snap(self, points_xy: np.ndarray, arc_ids: np.ndarray):
        """Distance (m) and fraction along the arc for aligned (point, arc) pairs."""
        return data.point_segment_dist(points_xy, self.a_xy[arc_ids], self.b_xy[arc_ids])

    # ---- connectivity -------------------------------------------------------
    def connected(self, a: int, b: int) -> bool:
        """True if b can directly follow a (same arc, or a ends where b starts)."""
        return a == b or self.arc_to[a] == self.arc_from[b]

    # ---- shortest paths -----------------------------------------------------
    def node_distances(self, src: int, limit: float) -> dict[int, float]:
        """Bounded Dijkstra from node ``src``: {node: metres} for all nodes within ``limit``."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        dist = {src: 0.0}
        done: dict[int, float] = {}
        heap = [(0.0, src)]
        indptr, nbr, nbr_w = self.indptr, self.nbr, self.nbr_w
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done[u] = d
            for k in range(indptr[u], indptr[u + 1]):
                v = int(nbr[k])
                nd = d + nbr_w[k]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[src] = (limit, done)
        return done

    def route_distance(self, a: int, fa: float, b: int, fb: float, limit: float,
                       backward_tol_m: float = 0.0) -> float:
        """Driving distance (m) from fraction ``fa`` along arc ``a`` to fraction ``fb`` along arc ``b``.

        Returns ``np.inf`` when b is not reachable within ``limit`` metres.
        A small backward move on the same arc (<= ``backward_tol_m``) is treated as GPS
        jitter of a stationary/slow car and costs 0 m instead of a loop around the block.
        """
        la = self.arc_len[a]
        if a == b:
            if fb >= fa:
                return (fb - fa) * la
            if (fa - fb) * la <= backward_tol_m:
                return 0.0
        rest_a = (1.0 - fa) * la
        head_b = fb * self.arc_len[b]
        budget = limit - rest_a - head_b
        if budget < 0:
            return np.inf
        d = self.node_distances(int(self.arc_to[a]), budget).get(int(self.arc_from[b]))
        return np.inf if d is None else rest_a + d + head_b


def route_continuity(pred_arcs, graph: RoadGraph) -> float:
    """Share of consecutive predicted pairs (ignoring missing = -1) that are directly connected.

    Note: at 1 Hz a car can skip a very short arc between two fixes, so even the
    reference labels can score slightly below 1.0 -- report that ceiling too.
    """
    p = np.asarray([x for x in pred_arcs if x is not None and x >= 0], dtype=np.int64)
    if len(p) < 2:
        return float("nan")
    a, b = p[:-1], p[1:]
    ok = (a == b) | (graph.arc_to[a] == graph.arc_from[b])
    return float(ok.mean())


def load_obs_labels(split: str):
    """A's per-point reference labels: {trip_id: DataFrame indexed by obs_idx}."""
    import pandas as pd
    path = ROOT / "results" / f"obs_labels_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python src/build_candidates.py` (role A) first")
    df = pd.read_csv(path, dtype={"trip_id": str})
    return {tid: g.set_index("obs_idx").sort_index() for tid, g in df.groupby("trip_id")}
