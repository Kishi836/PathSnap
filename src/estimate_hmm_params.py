"""Estimate initial HMM parameters from the TRAIN partition only (Newson & Krumm, 2009).

* sigma : 1.4826 * median(distance from GPS point to its reference arc)
          (robust estimate of the GPS noise standard deviation)
* beta  : median(|d_route - d_gc|) / ln 2 over consecutive 1-second point pairs
          on the reference path (the median of an exponential is beta * ln 2)

These are starting values; the decoder tuning experiment refines them.
Usage:  python -m src.estimate_hmm_params
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import ROOT, RoadGraph, haversine_m, list_trips, load_trip, point_to_segments
from .standins import reference_labels


def trip_stats(trip_id: str):
    trip = load_trip(trip_id, "train")
    g = RoadGraph(trip)
    ref = reference_labels(trip, g)
    gps = trip.gps_xy
    ok = ref >= 0
    d, f = point_to_segments(gps[:, 0], gps[:, 1], g.ax[np.maximum(ref, 0)], g.ay[np.maximum(ref, 0)],
                             g.bx[np.maximum(ref, 0)], g.by[np.maximum(ref, 0)])
    lon, lat, ts = trip.track.T
    diffs = []
    for t in range(len(ref) - 1):
        if not (ok[t] and ok[t + 1]) or ts[t + 1] - ts[t] != 1:
            continue
        d_gc = haversine_m(lon[t], lat[t], lon[t + 1], lat[t + 1])
        rd = g.route_distance(int(ref[t]), f[t], int(ref[t + 1]), f[t + 1], 2 * d_gc + 300, 10.0)
        if np.isfinite(rd):
            diffs.append(abs(rd - d_gc))
    return d[ok], np.asarray(diffs)


def main():
    dists, diffs = [], []
    for tid in list_trips("train"):
        d, df = trip_stats(tid)
        dists.append(d)
        diffs.append(df)
    d, df = np.concatenate(dists), np.concatenate(diffs)
    sigma = 1.4826 * np.median(d)
    beta = np.median(df) / np.log(2)
    out = pd.DataFrame([{
        "partition": "train", "n_trips": len(dists), "n_points": len(d), "n_pairs": len(df),
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
