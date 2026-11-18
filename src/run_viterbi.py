"""Run the HMM/Viterbi decoder on train / student_test trips and score it (role C).

Phase 1:  python -m src.run_viterbi --split train --trips 00000001
All dev:  python -m src.run_viterbi --split student_test --trips all

Inputs (uses A's files when they exist, otherwise the stand-ins in src/standins.py):
  results/candidates_{split}.parquet,  results/obs_labels_{split}.csv
Outputs:
  results/viterbi/{split}_{trip}_points.csv   per-point predictions + diagnostics
  results/viterbi_runs.csv                    one row per trip per run (appended)
  results/maps/{split}_{trip}.html            folium map
  models/hmm_config.json                      decoder params for D
Never run on the evaluator partition: the reference route is used here for scoring.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .maps import trip_map
from .network import ROOT, RoadGraph, list_trips, load_trip, route_continuity
from .standins import reference_labels, simple_candidates
from .viterbi import HMMParams, decode, top1

RESULTS = ROOT / "results"
ALLOWED_SPLITS = ("train", "student_test")


def default_params() -> HMMParams:
    """HMM params: estimates from train (src/estimate_hmm_params.py) if available."""
    p = HMMParams()
    est = RESULTS / "hmm_param_estimates.csv"
    if est.exists():
        row = pd.read_csv(est).iloc[0]
        p.sigma_m, p.beta_m = round(float(row.sigma_m), 2), round(float(row.beta_m), 2)
    return p


def load_candidates(trip, graph, ref, split):
    path = RESULTS / f"candidates_{split}.parquet"
    if path.exists():
        df = pd.read_parquet(path, filters=[("trip_id", "==", trip.trip_id)])
        return df, "A"
    return simple_candidates(trip, graph, ref), "standin"


def load_obs_labels(trip, graph, split):
    path = RESULTS / f"obs_labels_{split}.csv"
    if path.exists():
        df = pd.read_csv(path, dtype={"trip_id": str})
        df = df[df.trip_id == trip.trip_id].set_index("obs_idx")["ref_arc_id"]
        ref = np.full(len(trip.track), -1, np.int64)
        ok = df.notna()
        ref[df.index[ok].to_numpy()] = df[ok].astype(np.int64).to_numpy()
        return ref, "A"
    return reference_labels(trip, graph), "standin"


def jaccard(a, b):
    a, b = set(int(x) for x in a if x >= 0), set(int(x) for x in b if x >= 0)
    return len(a & b) / max(1, len(a | b))


def run_trip(trip_id, split, params, make_map=True):
    trip = load_trip(trip_id, split)
    graph = RoadGraph(trip)
    ref, label_src = load_obs_labels(trip, graph, split)
    cands, cand_src = load_candidates(trip, graph, ref, split)
    T = len(trip.track)
    has_ref = ref >= 0

    # baseline "classifier-only" stand-in: nearest candidate arc per point
    dist_col = "f_dist_m" if "f_dist_m" in cands else None
    score = -cands[dist_col].to_numpy() if dist_col else None
    t0 = time.perf_counter()
    pred, info = decode(cands, None, graph, params, return_info=True)
    t_dec = time.perf_counter() - t0
    if score is None:  # recompute distance if A's table names it differently
        from .viterbi import _snap
        score = -_snap(graph, trip.gps_xy, cands.obs_idx.to_numpy(), cands.arc_id.to_numpy())[0]
    base = top1(cands, score, T)

    lab = cands[cands.label == 1].obs_idx.nunique() if "label" in cands else np.nan
    row = {
        "run_at": datetime.now().isoformat(timespec="seconds"), "split": split, "trip_id": trip_id,
        "emission": "distance", "candidates_from": cand_src, "labels_from": label_src,
        **{k: v for k, v in vars(params).items()},
        "n_points": T, "n_ref_points": int(has_ref.sum()),
        "candidate_miss_rate": 1 - lab / max(1, has_ref.sum()),
        "points_without_candidates": int((info.n_candidates == 0).sum()),
        "acc_top1": float(np.mean(base[has_ref] == ref[has_ref])),
        "acc_viterbi": float(np.mean(pred[has_ref] == ref[has_ref])),
        "continuity_top1": route_continuity(base, graph),
        "continuity_viterbi": route_continuity(pred, graph),
        "continuity_reference": route_continuity(ref[has_ref], graph),
        "arc_jaccard_top1": jaccard(base, trip.route),
        "arc_jaccard_viterbi": jaccard(pred, trip.route),
        "segments": int(info.segment.max() + 1), "forced_breaks": int(info.forced_break.sum()),
        "decode_seconds": round(t_dec, 3),
    }

    out = info.assign(lon=trip.track[:, 0], lat=trip.track[:, 1], t=trip.track[:, 2],
                      ref_arc=ref, top1_arc=base)
    (RESULTS / "viterbi").mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "viterbi" / f"{split}_{trip_id}_points.csv", index=False)
    if make_map:
        title = (f"{split} {trip_id} — Viterbi acc {row['acc_viterbi']:.1%} "
                 f"(nearest-arc {row['acc_top1']:.1%})")
        trip_map(graph, pred, ref, title=title, out_path=RESULTS / "maps" / f"{split}_{trip_id}.html")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=ALLOWED_SPLITS)
    ap.add_argument("--trips", nargs="+", default=["00000001"], help="trip ids or 'all'")
    ap.add_argument("--sigma", type=float)
    ap.add_argument("--beta", type=float)
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--save-config", action="store_true", help="write models/hmm_config.json")
    a = ap.parse_args()

    params = default_params()
    if a.sigma:
        params.sigma_m = a.sigma
    if a.beta:
        params.beta_m = a.beta
    trips = list_trips(a.split) if a.trips == ["all"] else a.trips

    rows = []
    for tid in trips:
        r = run_trip(tid, a.split, params, make_map=not a.no_map)
        rows.append(r)
        print(f"{a.split} {tid}: top1 {r['acc_top1']:.3f} -> viterbi {r['acc_viterbi']:.3f} | "
              f"continuity {r['continuity_top1']:.3f} -> {r['continuity_viterbi']:.3f} "
              f"(ref {r['continuity_reference']:.3f}) | miss {r['candidate_miss_rate']:.3f} | "
              f"{r['decode_seconds']:.1f}s")
    df = pd.DataFrame(rows)
    log = RESULTS / "viterbi_runs.csv"
    df.to_csv(log, mode="a", header=not log.exists(), index=False)
    if len(df) > 1:
        print(df[["acc_top1", "acc_viterbi", "continuity_top1", "continuity_viterbi"]].mean().round(4).to_string())
    if a.save_config:
        params.to_json(ROOT / "models" / "hmm_config.json", emission="distance",
                       candidate_radius_m=50.0, candidate_k=8,
                       note="sigma/beta estimated on train (src/estimate_hmm_params.py); not yet tuned")
        print("saved models/hmm_config.json")


if __name__ == "__main__":
    main()
