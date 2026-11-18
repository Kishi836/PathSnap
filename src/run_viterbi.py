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
from .viterbi import HMMParams, classifier_emission_logp, decode, top1

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


def load_candidates(split: str):
    path = RESULTS / f"candidates_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python src/build_candidates.py` (role A) first")
    df = pd.read_parquet(path)   # all columns: the classifier emission needs the f_* features
    return {tid: g.reset_index(drop=True) for tid, g in df.groupby("trip_id")}


def jaccard(a, b):
    a, b = set(int(x) for x in a if x >= 0), set(int(x) for x in b if x >= 0)
    return len(a & b) / max(1, len(a | b))


def run_trip(trip_id, split, params, cands, labels, make_map=True, prob=None):
    """prob: classifier P(candidate is the true arc) per row of cands; None = distance-only emissions."""
    trip = load_trip(trip_id, split)
    graph = RoadGraph(trip)
    T = len(trip.track)
    ref = labels["ref_arc_id"].reindex(range(T)).fillna(-1).to_numpy(np.int64)
    has_ref = ref >= 0
    moving = has_ref & ~labels["stationary"].reindex(range(T)).fillna(False).to_numpy(bool)
    # points whose reference arc is among the candidates (B trains/evaluates on these)
    hit = np.zeros(T, bool)
    hit[cands.loc[cands.label == 1, "obs_idx"].to_numpy()] = True

    emission = None if prob is None else params.emission_weight * classifier_emission_logp(prob)
    t0 = time.perf_counter()
    pred, info = decode(cands, emission, graph, params, return_info=True)
    t_dec = time.perf_counter() - t0
    # "classifier-only": per-point top-1 of the same scores the HMM uses (nearest arc for distance-only)
    base = top1(cands, -cands["f_dist"].to_numpy() if prob is None else prob, T)

    row = {
        "run_at": datetime.now().isoformat(timespec="seconds"), "split": split, "trip_id": trip_id,
        "emission": "distance" if prob is None else "classifier", "candidates_from": "A", "labels_from": "A",
        **{k: v for k, v in vars(params).items()},
        "n_points": T, "n_ref_points": int(has_ref.sum()),
        "candidate_miss_rate": float(1 - hit[has_ref].mean()),
        "points_without_candidates": int((info.n_candidates == 0).sum()),
        "acc_top1": float(np.mean(base[has_ref] == ref[has_ref])),
        "acc_viterbi": float(np.mean(pred[has_ref] == ref[has_ref])),
        "acc_top1_on_hits": float(np.mean(base[hit] == ref[hit])),
        "acc_viterbi_on_hits": float(np.mean(pred[hit] == ref[hit])),
        "n_moving_points": int(moving.sum()),
        "acc_top1_moving": float(np.mean(base[moving] == ref[moving])) if moving.any() else np.nan,
        "acc_viterbi_moving": float(np.mean(pred[moving] == ref[moving])) if moving.any() else np.nan,
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
        source = "MLP alone" if prob is not None else "nearest arc"
        title = (f"{split} {trip_id}, moving fixes: HMM {row['acc_viterbi_moving']:.1%} "
                 f"({source} {row['acc_top1_moving']:.1%})")
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
    ap.add_argument("--emission", default="distance", choices=("distance", "classifier"),
                    help="classifier = log P from models/classifier.joblib (B's model)")
    ap.add_argument("--emission-weight", type=float)
    ap.add_argument("--backward-tol", type=float)
    a = ap.parse_args()

    model = cfg = None
    if a.emission == "classifier":
        import json
        import joblib
        model = joblib.load(ROOT / "models" / "classifier.joblib")
        cfg = json.loads((ROOT / "models" / "classifier_config.json").read_text())

    params = default_params()
    if a.sigma:
        params.sigma_m = a.sigma
    if a.beta:
        params.beta_m = a.beta
    if a.emission_weight:
        params.emission_weight = a.emission_weight
    if a.backward_tol:
        params.backward_tol_m = a.backward_tol
    trips = list_trips(a.split) if a.trips == ["all"] else a.trips
    cands, labels = load_candidates(a.split), load_obs_labels(a.split)

    rows = []
    for tid in trips:
        prob = None if model is None else model.predict_proba(cands[tid][cfg["feature_columns"]])[:, 1]
        r = run_trip(tid, a.split, params, cands[tid], labels[tid], make_map=not a.no_map, prob=prob)
        rows.append(r)
        print(f"{a.split} {tid}: top1 {r['acc_top1']:.3f} -> viterbi {r['acc_viterbi']:.3f} "
              f"(moving {r['acc_top1_moving']:.3f} -> {r['acc_viterbi_moving']:.3f}) | "
              f"continuity {r['continuity_top1']:.3f} -> {r['continuity_viterbi']:.3f} "
              f"(ref {r['continuity_reference']:.3f}) | miss {r['candidate_miss_rate']:.3f} | "
              f"{r['decode_seconds']:.1f}s")
    df = pd.DataFrame(rows)
    log = RESULTS / "viterbi_runs.csv"
    # rewrite instead of appending, so rows with new columns never land under an old header
    if log.exists():
        df = pd.concat([pd.read_csv(log, dtype={"trip_id": str}), df], ignore_index=True)
    df.to_csv(log, index=False)
    df = df.tail(len(rows))
    if len(df) > 1:
        w = df.n_ref_points / df.n_ref_points.sum()
        cols = ["candidate_miss_rate", "acc_top1", "acc_viterbi", "acc_viterbi_on_hits",
                "continuity_top1", "continuity_viterbi"]
        print("point-weighted mean:", (df[cols].mul(w, axis=0).sum()).round(4).to_dict())
        wm = df.n_moving_points / df.n_moving_points.sum()
        print("moving points only:", (df[["acc_top1_moving", "acc_viterbi_moving"]].mul(wm, axis=0).sum()).round(4).to_dict())
    if a.save_config:
        params.to_json(ROOT / "models" / "hmm_config.json", emission="distance",
                       candidates="A src/build_candidates.py trip_features (radius 50 m, all arcs)",
                       note="sigma/beta estimated on train (src/estimate_hmm_params.py); not yet tuned")
        print("saved models/hmm_config.json")


if __name__ == "__main__":
    main()
