"""Check the tuned decoder settings on the 80 train trips, which were not used to choose them.

Classifier probabilities are out-of-fold (5-fold GroupKFold by trip, tuned MLP refit per fold), so no
train trip is scored by a model that saw it. Run from the repo root:  python -m src.validate_decoder
Writes results/decoder_validation_train.csv (one row per setting per trip) and prints point-weighted totals.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .network import ROOT
from .run_viterbi import RESULTS, default_params, load_candidates, load_obs_labels, run_trip

sys.path.insert(0, str(Path(__file__).parent))
import train_models as tm                       # noqa: E402  (script-style module in src/)
from evaluate_tuned import pipeline_params      # noqa: E402

SETTINGS = {
    "default decoder": dict(beta_m=1.75, emission_weight=1.0, backward_tol_m=10.0),
    "tuned decoder": dict(beta_m=3.0, emission_weight=0.5, backward_tol_m=30.0),
}
_JOB = {}


def _init(cands, labels, probs):
    _JOB.update(cands=cands, labels=labels, probs=probs)


def _run(task):
    name, tid = task
    params = default_params()
    for k, v in SETTINGS[name].items():
        setattr(params, k, v)
    r = run_trip(tid, "train", params, _JOB["cands"][tid], _JOB["labels"][tid], make_map=False, prob=_JOB["probs"][tid])
    return {"setting": name, "trip_id": tid, **{k: r[k] for k in (
        "n_ref_points", "n_moving_points", "acc_top1", "acc_viterbi", "acc_top1_moving", "acc_viterbi_moving",
        "continuity_viterbi")}}


def out_of_fold_probs():
    cfg = json.loads((ROOT / "models" / "classifier_config.json").read_text())
    full = pd.read_parquet(tm.TRAIN_PATH)
    fit_rows = tm.training_subset(full)
    trips = np.array(sorted(full.trip_id.unique()))
    prob = np.empty(len(full))
    for k, (tr, te) in enumerate(GroupKFold(n_splits=tm.N_SPLITS).split(trips, groups=trips)):
        fit = fit_rows[fit_rows.trip_id.isin(trips[tr])]
        model = tm.create_models()["MLP"].set_params(**pipeline_params(cfg["best_params"]))
        model.fit(fit[cfg["feature_columns"]], fit.label)
        held = full.trip_id.isin(trips[te]).to_numpy()
        prob[held] = model.predict_proba(full.loc[held, cfg["feature_columns"]])[:, 1]
        print(f"fold {k + 1}: {len(te)} held-out trips", flush=True)
    full["prob"] = prob
    return {tid: g.prob.to_numpy() for tid, g in full.groupby("trip_id")}


def main():
    probs = out_of_fold_probs()
    cands, labels = load_candidates("train"), load_obs_labels("train")
    tasks = [(name, tid) for name in SETTINGS for tid in cands]
    os.environ["OMP_NUM_THREADS"] = "1"
    with Pool(10, initializer=_init, initargs=(cands, labels, probs)) as pool:
        rows = pool.map(_run, tasks, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "decoder_validation_train.csv", index=False)
    for name, g in df.groupby("setting", sort=False):
        w, wm = g.n_ref_points, g.n_moving_points
        print(f"{name}: MLP alone {np.average(g.acc_top1_moving, weights=wm):.4f} -> HMM "
              f"{np.average(g.acc_viterbi_moving, weights=wm):.4f} (moving) | all points "
              f"{np.average(g.acc_viterbi, weights=w):.4f}")


if __name__ == "__main__":
    main()
