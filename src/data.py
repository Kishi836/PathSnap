"""Raw-data loading and geometry helpers shared by EDA, feature engineering and the API."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "Student folder" / "Student folder"
PARTITIONS = ("train", "student_test")
EARTH_R = 6_371_000.0  # metres


def trip_ids(partition):
    return sorted(p.name for p in (DATA_ROOT / partition).iterdir() if p.is_dir())


def _read(path, names, dtype="float64"):
    # whitespace-separated, no header; keep NaNs visible instead of crashing
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def load_trip(partition, trip_id):
    """Return dict of DataFrames: track, nodes, arcs, route (route is None if file absent)."""
    base = DATA_ROOT / partition / trip_id / trip_id
    route_path = base.with_suffix(".route")
    return {
        "track": _read(base.with_suffix(".track"), ["lon", "lat", "t"]),
        "nodes": _read(base.with_suffix(".nodes"), ["lon", "lat"]),
        "arcs": _read(base.with_suffix(".arcs"), ["u", "v"], dtype="Int64"),
        "route": _read(route_path, ["arc_id"], dtype="Int64") if route_path.exists() else None,
    }


class LocalProjection:
    """Equirectangular projection to metres around a reference latitude.

    Trips span tens of km at ~49.8N, so the scale error vs a proper UTM projection is
    well under 1% — negligible for GPS noise of a few metres — and needs no extra dependency.
    """

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = np.radians(lat0), np.radians(lon0)
        self.kx = EARTH_R * np.cos(self.lat0)

    def __call__(self, lon, lat):
        x = self.kx * (np.radians(lon) - self.lon0)
        y = EARTH_R * (np.radians(lat) - self.lat0)
        return np.column_stack([x, y])


def point_segment_dist(p, a, b):
    """Distance (m) from points p to segments a->b; inputs broadcast, last dim = 2 (x, y).

    Use aligned (k,2) arrays for k point/segment pairs, or p[:, None] with a[None], b[None]
    for an all-pairs (n, m) matrix. Also returns t in [0,1]: where along a->b the closest point is.
    """
    ab = b - a
    denom = np.maximum((ab ** 2).sum(-1), 1e-12)
    t = np.clip(((p - a) * ab).sum(-1) / denom, 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1), t


def network_xy(trip, proj):
    """Projected node coordinates and arc endpoint arrays (a, b) in metres."""
    nodes = proj(trip["nodes"].lon.to_numpy(), trip["nodes"].lat.to_numpy())
    arcs = trip["arcs"].to_numpy(dtype=np.int64)
    return nodes, nodes[arcs[:, 0]], nodes[arcs[:, 1]]


def stationary_mask(xy, t, half_window_s=15, max_disp_m=20):
    """True where the vehicle is parked: net displacement across ±half_window_s is under max_disp_m.

    Instantaneous speed can't detect this — GPS jitter while parked shows up as 1–3 m/s.
    A car moving at even 5 km/h covers ~40 m in 30 s, so 20 m separates the two cleanly.
    """
    lo = np.searchsorted(t, t - half_window_s)
    hi = np.searchsorted(t, t + half_window_s, side="right") - 1
    return np.linalg.norm(xy[hi] - xy[lo], axis=1) < max_disp_m


def arcs_within(points, a, b, radius, long_half=150.0):
    """All (point, arc) pairs with point-to-arc distance <= radius.

    KD-tree on arc midpoints: an arc can only be within `radius` of a point if its midpoint is
    within radius + half its length. Arcs longer than 2*long_half (≈1% of arcs) would blow up
    that search radius, so they are checked brute-force instead.
    Returns (point_idx, arc_idx, dist, t) arrays.
    """
    from scipy.spatial import KDTree

    half = np.linalg.norm(b - a, axis=1) / 2
    short = np.flatnonzero(half <= long_half)
    long_ = np.flatnonzero(half > long_half)

    tree = KDTree(((a + b) / 2)[short])
    hits = tree.query_ball_point(points, r=radius + long_half)
    pi = np.repeat(np.arange(len(points)), [len(h) for h in hits])
    ai = short[np.concatenate(hits).astype(np.int64)] if len(pi) else np.empty(0, np.int64)
    if len(long_):
        pi = np.concatenate([pi, np.repeat(np.arange(len(points)), len(long_))])
        ai = np.concatenate([ai, np.tile(long_, len(points))])

    d, t = point_segment_dist(points[pi], a[ai], b[ai])
    keep = d <= radius
    return pi[keep], ai[keep], d[keep], t[keep]


if __name__ == "__main__":
    # self-check: point (0,1) is 1 m from segment (−1,0)->(1,0), projecting to its middle
    d, t = point_segment_dist(np.array([[0.0, 1.0]]), np.array([[-1.0, 0.0]]), np.array([[1.0, 0.0]]))
    assert np.isclose(d[0], 1.0) and np.isclose(t[0], 0.5)
    # beyond the end: clamps to endpoint
    d, t = point_segment_dist(np.array([[3.0, 0.0]]), np.array([[-1.0, 0.0]]), np.array([[1.0, 0.0]]))
    assert np.isclose(d[0], 2.0) and t[0] == 1.0
    # arcs_within must agree with brute force, including a long arc
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 2000, (300, 2)); B = A + rng.normal(0, 40, (300, 2)); B[0] = A[0] + [1500, 0]
    P = rng.uniform(0, 2000, (50, 2))
    pi, ai, _, _ = arcs_within(P, A, B, radius=60)
    brute = np.argwhere(point_segment_dist(P[:, None], A[None], B[None])[0] <= 60)
    assert set(zip(pi, ai)) == set(map(tuple, brute))
    # parked (jitter around origin) vs driving at 10 m/s
    t = np.arange(100.0)
    parked = rng.normal(0, 3, (100, 2))
    driving = np.column_stack([t * 10, np.zeros(100)])
    assert stationary_mask(parked, t).all() and not stationary_mask(driving, t).any()
    # 0.001 deg latitude ~ 111.2 m
    proj = LocalProjection(49.85, 8.6)
    assert abs(proj(np.array([8.6]), np.array([49.851]))[0, 1] - 111.19) < 0.1
    print("data.py self-check OK")
