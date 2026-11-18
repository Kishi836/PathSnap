"""No-ML baseline: snap every GPS point to its nearest candidate arc.

Run from the repo root:  python src/baseline_nearest.py
Writes results/baseline_nearest.csv (per trip) and prints partition totals.
Any classifier or decoder should beat these numbers.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import PARTITIONS, load_trip

RESULTS = Path(__file__).resolve().parents[1] / "results"


def route_continuity(pred, arcs):
    """Fraction of consecutive predictions that stay on the same arc or move to a directly connected one."""
    p, q = pred[:-1], pred[1:]
    ok = (p >= 0) & (q >= 0)
    p, q = p[ok], q[ok]
    return np.mean((p == q) | (arcs[p, 1] == arcs[q, 0])) if len(p) else np.nan


def evaluate(partition):
    cands = pd.read_parquet(RESULTS / f"candidates_{partition}.parquet", columns=["trip_id", "obs_idx", "arc_id", "f_dist_rank"])
    labels = pd.read_csv(RESULTS / f"obs_labels_{partition}.csv", dtype={"trip_id": str})
    nearest = cands[cands.f_dist_rank == 0].set_index(["trip_id", "obs_idx"]).arc_id
    rows = []
    for tid, lab in labels.groupby("trip_id", sort=True):
        pred = nearest.reindex(pd.MultiIndex.from_arrays([lab.trip_id, lab.obs_idx])).fillna(-1).to_numpy(np.int64)
        correct = pred == lab.ref_arc_id.to_numpy()
        moving = ~lab.stationary.to_numpy()
        arcs = load_trip(partition, tid)["arcs"].to_numpy(dtype=np.int64)
        # direction-blind check: count as correct if the prediction is the same road in either direction
        ref = arcs[lab.ref_arc_id.to_numpy()]
        has = pred >= 0
        same_road = np.zeros(len(pred), bool)
        same_road[has] = (np.sort(arcs[pred[has]], axis=1) == np.sort(ref[has], axis=1)).all(1)
        rows.append({
            "partition": partition, "trip_id": tid, "n_points": len(lab),
            "top1_acc": correct.mean(),
            "top1_acc_ignore_direction": same_road.mean(),
            "top1_acc_moving": correct[moving].mean() if moving.any() else np.nan,
            "no_prediction_frac": np.mean(pred < 0),
            "route_continuity": route_continuity(pred, arcs),
        })
    return rows


if __name__ == "__main__":
    # self-check: arcs 0:(0->1), 1:(1->2), 2:(5->6). Pairs: 0-0 same, 0-1 connected, 1-2 not; pairs with -1 skipped
    arcs = np.array([[0, 1], [1, 2], [5, 6]])
    assert route_continuity(np.array([0, 0, 1, 2, -1, 2]), arcs) == 2 / 3

    df = pd.DataFrame([r for p in PARTITIONS for r in evaluate(p)])
    df.to_csv(RESULTS / "baseline_nearest.csv", index=False)
    for part, g in df.groupby("partition", sort=False):
        w = g.n_points / g.n_points.sum()
        print(f"{part}: point-weighted top-1 {np.sum(w * g.top1_acc):.4f} | ignoring direction {np.sum(w * g.top1_acc_ignore_direction):.4f}"
              f" | moving only {np.sum(w * g.top1_acc_moving):.4f}"
              f" | route continuity {np.sum(w * g.route_continuity):.4f} | per-trip mean top-1 {g.top1_acc.mean():.4f}")
