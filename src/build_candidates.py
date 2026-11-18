"""Per-point reference labels, candidate generation and feature engineering.

Run from the repo root:  python src/build_candidates.py
Writes: results/obs_labels_{partition}.csv, results/candidates_{partition}.parquet,
        results/candidates_summary.csv

`trip_features()` never reads the reference route, so the API can call it at inference time.
Labels are attached afterwards from `route_labels()`, which is the only code that reads `.route`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import (PARTITIONS, LocalProjection, arcs_within, load_trip, network_xy, point_segment_dist,
                  stationary_mask, trip_ids)

OUT = Path(__file__).resolve().parents[1] / "results"
RADIUS_M = 50.0          # EDA: route within 50 m for 99.9% of student_test points, ~12 candidates/point
HEADING_WINDOW = 2       # GPS heading from point i-2 to i+2
MIN_HEADING_DISP_M = 5.0 # below this displacement the heading is noise


def route_labels(xy, a, b, route):
    """Assign every GPS point a position along the reference route, never moving backwards.

    Dynamic programming: cost[i, k] = dist(point i, route arc k) + min_{k' <= k} cost[i-1, k'].
    Minimising total distance under the "monotone along the route" constraint fixes the two
    failure modes of plain nearest-arc matching seen in EDA: back-steps (~2% of points) and
    loops where the route passes the same place twice.
    Returns (route_position per point, distance to that arc in m).
    """
    ra, rb = a[route], b[route]
    n, m = len(xy), len(route)
    assert m < 2 ** 16
    back = np.empty((n, m), dtype=np.uint16)
    idx = np.arange(m)
    cost, _ = point_segment_dist(xy[0][None], ra, rb)
    back[0] = idx
    for i in range(1, n):
        d, _ = point_segment_dist(xy[i][None], ra, rb)
        run_min = np.minimum.accumulate(cost)
        # latest position <= k where the running minimum was achieved
        back[i] = np.maximum.accumulate(np.where(cost == run_min, idx, 0))
        cost = d + run_min
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    dist, _ = point_segment_dist(xy, ra[pos], rb[pos])
    return pos, dist


def gps_heading(xy):
    """Unit direction of travel per point (from i-W to i+W) and whether it is reliable."""
    n = len(xy)
    lo = np.clip(np.arange(n) - HEADING_WINDOW, 0, n - 1)
    hi = np.clip(np.arange(n) + HEADING_WINDOW, 0, n - 1)
    v = xy[hi] - xy[lo]
    norm = np.linalg.norm(v, axis=1)
    valid = norm >= MIN_HEADING_DISP_M
    return v / np.maximum(norm, 1e-9)[:, None], valid


def trip_features(trip):
    """Candidate (point, arc) pairs with features for one trip. Uses track + network only."""
    track = trip["track"]
    proj = LocalProjection(track.lat.mean(), track.lon.mean())
    xy = proj(track.lon.to_numpy(), track.lat.to_numpy())
    t = track.t.to_numpy()
    _, a, b = network_xy(trip, proj)
    arcs = trip["arcs"].to_numpy(dtype=np.int64)

    pi, ai, dist, tpos = arcs_within(xy, a, b, RADIUS_M)
    order = np.lexsort((dist, pi))                 # group by point, nearest first
    pi, ai, dist, tpos = pi[order], ai[order], dist[order], tpos[order]

    n = len(xy)
    n_cands = np.bincount(pi, minlength=n)
    first = np.r_[0, np.cumsum(n_cands)[:-1]]
    rank = np.arange(len(pi)) - first[pi]
    min_dist = dist[first[pi]]

    # direction agreement: two-way roads give two arcs with identical geometry, only heading separates them
    heading, h_valid = gps_heading(xy)
    arc_vec = b[ai] - a[ai]
    arc_unit = arc_vec / np.maximum(np.linalg.norm(arc_vec, axis=1), 1e-9)[:, None]
    heading_cos = np.where(h_valid[pi], (heading[pi] * arc_unit).sum(1), 0.0)

    # temporal context: the true arc is usually close to the neighbouring GPS points as well
    prev_i, next_i = np.maximum(pi - 1, 0), np.minimum(pi + 1, n - 1)
    dist_prev, _ = point_segment_dist(xy[prev_i], a[ai], b[ai])
    dist_next, _ = point_segment_dist(xy[next_i], a[ai], b[ai])

    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.diff(t)
    speed = np.r_[0.0, step / np.maximum(dt, 1e-9)]
    gap_before = np.r_[0.0, dt]

    pairs = set(map(tuple, arcs))
    two_way = np.array([(v, u) in pairs for u, v in arcs])
    out_deg = np.bincount(arcs[:, 0], minlength=len(trip["nodes"]))

    return pd.DataFrame({
        "obs_idx": pi,
        "arc_id": ai,
        "f_dist": dist,
        "f_dist_rank": rank,
        "f_dist_minus_min": dist - min_dist,
        "f_n_cands": n_cands[pi],
        "f_heading_cos": heading_cos,
        "f_heading_valid": h_valid[pi].astype(np.int8),
        "f_dist_prev": dist_prev,
        "f_dist_next": dist_next,
        "f_proj_t": tpos,
        "f_arc_len": np.linalg.norm(arc_vec, axis=1),
        "f_arc_two_way": two_way[ai].astype(np.int8),
        "f_end_out_degree": out_deg[arcs[ai, 1]],
        "f_speed": speed[pi],
        "f_gap_before_s": gap_before[pi],
        "f_stationary": stationary_mask(xy, t)[pi].astype(np.int8),
    }), xy, a, b


def build(partition):
    cand_frames, label_frames, summary = [], [], []
    for tid in trip_ids(partition):
        trip = load_trip(partition, tid)
        feats, xy, a, b = trip_features(trip)
        route = trip["route"].arc_id.to_numpy(dtype=np.int64)
        pos, ref_dist = route_labels(xy, a, b, route)
        ref_arc = route[pos]
        stationary = stationary_mask(xy, trip["track"].t.to_numpy())

        labels = pd.DataFrame({"trip_id": tid, "obs_idx": np.arange(len(xy)), "ref_arc_id": ref_arc,
                               "route_pos": pos, "ref_dist_m": ref_dist, "stationary": stationary})
        feats["label"] = (feats.arc_id.to_numpy() == ref_arc[feats.obs_idx.to_numpy()]).astype(np.int8)
        feats.insert(0, "trip_id", tid)

        has_cand = np.bincount(feats.obs_idx, minlength=len(xy)) > 0
        hit = np.bincount(feats.obs_idx, weights=feats.label, minlength=len(xy)) > 0
        # how often the in-order label differs from simply taking the nearest route arc
        nearest_pos = point_segment_dist(xy[:, None], a[route][None], b[route][None])[0].argmin(1)
        moving = ~stationary
        summary.append({
            "partition": partition, "trip_id": tid, "n_points": len(xy), "n_candidate_rows": len(feats),
            "no_candidate_frac": 1 - has_cand.mean(),
            "candidate_miss_frac": 1 - hit.mean(),
            "candidate_miss_frac_moving": 1 - hit[moving].mean() if moving.any() else np.nan,
            "label_differs_from_nearest_frac": np.mean(route[nearest_pos] != ref_arc),
            "ref_dist_p50_m": np.median(ref_dist), "ref_dist_p99_m": np.percentile(ref_dist, 99),
        })
        cand_frames.append(feats)
        label_frames.append(labels)
        print(f"{partition}/{tid}: {len(feats)} rows, miss {1 - hit.mean():.3%}")

    cands = pd.concat(cand_frames, ignore_index=True)
    float_cols = cands.select_dtypes("float64").columns
    cands[float_cols] = cands[float_cols].astype(np.float32)
    cands.to_parquet(OUT / f"candidates_{partition}.parquet", index=False, compression="zstd")
    pd.concat(label_frames, ignore_index=True).to_csv(OUT / f"obs_labels_{partition}.csv", index=False)
    return summary


def _self_check():
    # straight route east, then back west 30 m north (a loop): in-order matching must follow it
    a = np.array([[0, 0], [100, 0], [100, 30], [0, 30]], float)
    b = np.array([[100, 0], [100, 30], [0, 30], [-100, 30]], float)
    route = np.array([0, 1, 2, 3])
    xy = np.array([[10, 3], [50, 12], [90, 4], [100, 15], [60, 28], [20, 16], [-50, 31]], float)
    pos, _ = route_labels(xy, a, b, route)
    assert list(pos) == [0, 0, 0, 1, 2, 2, 3], pos   # point 5 is nearer arc 0 but comes after arc 2
    assert np.all(np.diff(pos) >= 0)


if __name__ == "__main__":
    _self_check()
    OUT.mkdir(exist_ok=True)
    rows = [r for p in PARTITIONS for r in build(p)]
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "candidates_summary.csv", index=False)
    for part, g in s.groupby("partition", sort=False):
        w = g.n_points / g.n_points.sum()
        print(f"{part}: {g.n_candidate_rows.sum()} rows | candidate miss {np.sum(w * g.candidate_miss_frac):.3%}"
              f" | label differs from nearest {np.sum(w * g.label_differs_from_nearest_frac):.3%}")
