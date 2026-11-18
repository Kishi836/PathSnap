"""Run the HMM/Viterbi decoder on train / student_test trips and score it (role C).

Phase 1:  python -m src.run_viterbi --split train --trips 00000001 --save-config
All dev:  python -m src.run_viterbi --split student_test --trips all --no-map

Inputs (role A, from `python src/build_candidates.py`):
  results/candidates_{split}.parquet,  results/obs_labels_{split}.csv
Outputs:
  results/viterbi/{split}_{trip}_points.csv   per-point predictions + diagnostics
  results/viterbi_runs.csv                    one row per trip per run (appended)
  results/maps/{split}_{trip}.html            folium map
  models/hmm_config.json                      decoder params for D
Never run on the evaluator partition: the reference labels are used here for scoring.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .maps import trip_map
from .network import ROOT, RoadGraph, list_trips, load_obs_labels, load_trip, route_continuity
from .viterbi import HMMParams, decode, top1

RESULTS = ROOT / "results"
ALLOWED_SPLITS = ("train", "student_test")
CAND_COLS = ["trip_id", "obs_idx", "arc_id", "label", "f_dist"]


def default_params() -> HMMParams:
    """HMM params: estimates from train (src/estimate_hmm_params.py) if available."""
    p = HMMParams()
    est = RESULTS / "hmm_param_estimates.csv"
    if est.exists():
        row = pd.read_csv(est).iloc[0]
        p.sigma_m, p.beta_m = round(float(row.sigma_m), 2), round(float(row.beta_m), 2)
    return p


def load_candidates(split: str):
    path = RESULTS / f"candidates_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python src/build_candidates.py` (role A) first")
    df = pd.read_parquet(path, columns=CAND_COLS)
    return {tid: g.reset_index(drop=True) for tid, g in df.groupby("trip_id")}


def jaccard(a, b):
    a, b = set(int(x) for x in a if x >= 0), set(int(x) for x in b if x >= 0)
    return len(a & b) / max(1, len(a | b))


def run_trip(trip_id, split, params, cands, labels, make_map=True):
    trip = load_trip(trip_id, split)
    graph = RoadGraph(trip)
    T = len(trip.track)
    ref = labels["ref_arc_id"].reindex(range(T)).fillna(-1).to_numpy(np.int64)
    has_ref = ref >= 0
    # points whose reference arc is among the candidates (B trains/evaluates on these)
    hit = np.zeros(T, bool)
    hit[cands.loc[cands.label == 1, "obs_idx"].to_numpy()] = True

    t0 = time.perf_counter()
    pred, info = decode(cands, None, graph, params, return_info=True)
    t_dec = time.perf_counter() - t0
    # baseline "classifier-only" stand-in until B's model exists: nearest candidate arc
    base = top1(cands, -cands["f_dist"].to_numpy(), T)

    row = {
        "run_at": datetime.now().isoformat(timespec="seconds"), "split": split, "trip_id": trip_id,
        "emission": "distance", "candidates_from": "A", "labels_from": "A",
        **{k: v for k, v in vars(params).items()},
        "n_points": T, "n_ref_points": int(has_ref.sum()),
        "candidate_miss_rate": float(1 - hit[has_ref].mean()),
        "points_without_candidates": int((info.n_candidates == 0).sum()),
        "acc_top1": float(np.mean(base[has_ref] == ref[has_ref])),
        "acc_viterbi": float(np.mean(pred[has_ref] == ref[has_ref])),
        "acc_top1_on_hits": float(np.mean(base[hit] == ref[hit])),
        "acc_viterbi_on_hits": float(np.mean(pred[hit] == ref[hit])),
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
    cands, labels = load_candidates(a.split), load_obs_labels(a.split)

    rows = []
    for tid in trips:
        r = run_trip(tid, a.split, params, cands[tid], labels[tid], make_map=not a.no_map)
        rows.append(r)
        print(f"{a.split} {tid}: top1 {r['acc_top1']:.3f} -> viterbi {r['acc_viterbi']:.3f} | "
              f"continuity {r['continuity_top1']:.3f} -> {r['continuity_viterbi']:.3f} "
              f"(ref {r['continuity_reference']:.3f}) | miss {r['candidate_miss_rate']:.3f} | "
              f"{r['decode_seconds']:.1f}s")
    df = pd.DataFrame(rows)
    log = RESULTS / "viterbi_runs.csv"
    df.to_csv(log, mode="a", header=not log.exists(), index=False)
    if len(df) > 1:
        w = df.n_ref_points / df.n_ref_points.sum()
        cols = ["candidate_miss_rate", "acc_top1", "acc_viterbi", "acc_viterbi_on_hits",
                "continuity_top1", "continuity_viterbi"]
        print("point-weighted mean:", (df[cols].mul(w, axis=0).sum()).round(4).to_dict())
    if a.save_config:
        params.to_json(ROOT / "models" / "hmm_config.json", emission="distance",
                       candidates="A src/build_candidates.py trip_features (radius 50 m, all arcs)",
                       note="sigma/beta estimated on train (src/estimate_hmm_params.py); not yet tuned")
        print("saved models/hmm_config.json")


if __name__ == "__main__":
    main()
