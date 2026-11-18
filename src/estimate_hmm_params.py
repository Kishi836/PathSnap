"""Estimate initial HMM parameters from the TRAIN partition only (Newson & Krumm, 2009).

Uses A's per-point reference labels (results/obs_labels_train.csv).

* sigma : 1.4826 * median(distance from GPS point to its reference arc)
          (robust estimate of the GPS noise standard deviation)
* beta  : median(|d_route - d_gc|) / ln 2 over consecutive 1-second point pairs
          on the reference path (the median of an exponential is beta * ln 2)

Points more than MAX_REF_DIST_M from their reference arc are left out of the estimate:
in a few train trips (e.g. 00000054) the route does not cover the whole track, so
those labels say nothing about GPS noise.

These are starting values; the decoder tuning experiment refines them.
Usage:  python -m src.estimate_hmm_params
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import ROOT, RoadGraph, haversine_m, list_trips, load_obs_labels, load_trip

MAX_REF_DIST_M = 100.0


def trip_stats(trip_id: str, labels: pd.DataFrame):
    trip = load_trip(trip_id, "train")
    g = RoadGraph(trip)
    ref = labels["ref_arc_id"].to_numpy(np.int64)
    d, f = g.snap(trip.gps_xy, ref)
    ok = d <= MAX_REF_DIST_M
    lon, lat, ts = trip.track.T
    diffs = []
    for t in range(len(ref) - 1):
        if not (ok[t] and ok[t + 1]) or ts[t + 1] - ts[t] != 1:
            continue
        d_gc = haversine_m(lon[t], lat[t], lon[t + 1], lat[t + 1])
        rd = g.route_distance(int(ref[t]), f[t], int(ref[t + 1]), f[t + 1], 2 * d_gc + 300, 10.0)
        if np.isfinite(rd):
            diffs.append(abs(rd - d_gc))
    return d[ok], np.asarray(diffs), int((~ok).sum())


def main():
    labels = load_obs_labels("train")
    dists, diffs, dropped = [], [], 0
    for tid in list_trips("train"):
        d, df, n_drop = trip_stats(tid, labels[tid])
        dists.append(d)
        diffs.append(df)
        dropped += n_drop
    d, df = np.concatenate(dists), np.concatenate(diffs)
    sigma = 1.4826 * np.median(d)
    beta = np.median(df) / np.log(2)
    out = pd.DataFrame([{
        "partition": "train", "labels": "A obs_labels_train.csv", "n_trips": len(dists),
        "n_points": len(d), "n_points_dropped_far": dropped, "n_pairs": len(df),
        "median_dist_to_ref_m": np.median(d), "p95_dist_to_ref_m": np.percentile(d, 95),
        "sigma_m": sigma, "median_abs_route_minus_gc_m": np.median(df), "beta_m": beta,
    }])
    path = ROOT / "results" / "hmm_param_estimates.csv"
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    print(out.T.to_string(header=False))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
