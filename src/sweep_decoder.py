"""Grid search of the HMM decoder parameters with classifier emissions (train / student_test only).

Run from the repo root:  python -m src.sweep_decoder [--split student_test]
Writes results/decoder_sweep_<split>.csv (one row per setting, point-weighted over the split's trips).
"""
import argparse
import itertools
import json
import os
from multiprocessing import Pool

import joblib
import pandas as pd

from .network import ROOT
from .run_viterbi import ALLOWED_SPLITS, RESULTS, default_params, load_candidates, load_obs_labels, run_trip

GRID = {
    "beta_m": [1.0, 1.75, 3.0, 5.0, 10.0],          # transition scale
    "emission_weight": [0.25, 0.5, 1.0, 2.0],       # classifier log P vs transitions
    # same-arc backward jitter treated as standing still; first sweep's best (30) sat on the grid edge, so extended
    "backward_tol_m": [10.0, 30.0, 60.0, 120.0],
}

_JOB = {}


def _init(split, cands, labels, probs):
    _JOB.update(split=split, cands=cands, labels=labels, probs=probs)


def evaluate(setting):
    params = default_params()
    for k, v in setting.items():
        setattr(params, k, v)
    rows = [run_trip(tid, _JOB["split"], params, _JOB["cands"][tid], _JOB["labels"][tid],
                     make_map=False, prob=_JOB["probs"][tid]) for tid in _JOB["cands"]]
    df = pd.DataFrame(rows)
    w, wm = df.n_ref_points, df.n_moving_points
    return {**setting,
            "acc_viterbi": (df.acc_viterbi * w).sum() / w.sum(),
            "acc_viterbi_moving": (df.acc_viterbi_moving * wm).sum() / wm.sum(),
            "continuity_viterbi": (df.continuity_viterbi * w).sum() / w.sum(),
            "acc_classifier_only": (df.acc_top1 * w).sum() / w.sum()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="student_test", choices=ALLOWED_SPLITS)
    ap.add_argument("--workers", type=int, default=10)
    split = ap.parse_args().split
    a = ap.parse_args()

    model = joblib.load(ROOT / "models" / "classifier.joblib")
    cfg = json.loads((ROOT / "models" / "classifier_config.json").read_text())
    cands, labels = load_candidates(split), load_obs_labels(split)
    probs = {tid: model.predict_proba(c[cfg["feature_columns"]])[:, 1] for tid, c in cands.items()}

    settings = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    os.environ["OMP_NUM_THREADS"] = "1"
    with Pool(a.workers, initializer=_init, initargs=(split, cands, labels, probs)) as pool:
        results = pool.map(evaluate, settings)

    out = pd.DataFrame(results).sort_values("acc_viterbi", ascending=False)
    out.to_csv(RESULTS / f"decoder_sweep_{split}.csv", index=False)
    print(f"{len(out)} settings on {split}, classifier: {cfg['model']}")
    print(out.head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
