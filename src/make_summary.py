"""Build results/summary.html: a one-page progress summary generated only from result files in the repo.

Run from the repo root:  python src/make_summary.py
Every number is read from results/*.csv or computed from results/*.parquet; plots are embedded as images.
"""
import base64
import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"


def img(path, alt):
    data = base64.b64encode((R / path).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def table(df, fmt):
    head = "".join(f"<th>{html.escape(c)}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(fmt.get(c, str)(v))}</td>" for c, v in r.items()) + "</tr>"
                   for _, r in df.iterrows())
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def wmean(values, weights):
    return float(np.average(values, weights=weights))


pct = lambda v: f"{100 * v:.1f}%"
f3 = lambda v: f"{v:.3f}"

# ---- data ----
inv = pd.read_csv(R / "eda/trip_inventory.csv", dtype={"trip_id": str})
tr = inv[inv.partition == "train"]
data_rows = []
for part, g in inv.groupby("partition", sort=False):
    data_rows.append({"Partition": part, "Trips": len(g), "GPS points": int(g.n_points.sum()),
                      "Steps exactly 1 s apart": wmean(g.frac_dt_eq_1, g.n_points - 1),
                      "Parked points": wmean(g.stationary_frac, g.n_points),
                      "Trips with a gap > 30 s": int((g.gaps_gt_30s > 0).sum())})
data_df = pd.DataFrame(data_rows)

cs = pd.read_csv(R / "candidates_summary.csv", dtype={"trip_id": str})
cand_rows = []
for part, g in cs.groupby("partition", sort=False):
    cand_rows.append({"Partition": part, "Candidate rows": int(g.n_candidate_rows.sum()),
                      "Candidates per point": g.n_candidate_rows.sum() / g.n_points.sum(),
                      "Candidate miss rate": wmean(g.candidate_miss_frac, g.n_points),
                      "Label differs from nearest route arc": wmean(g.label_differs_from_nearest_frac, g.n_points)})
cand_df = pd.DataFrame(cand_rows)

# two-way roads: how often the two nearest candidates are exactly tied in distance (student_test)
c = pd.read_parquet(R / "candidates_student_test.parquet", columns=["trip_id", "obs_idx", "f_dist_rank", "f_dist"])
top2 = c[c.f_dist_rank <= 1].pivot_table(index=["trip_id", "obs_idx"], columns="f_dist_rank", values="f_dist").dropna()
tie_frac = float(((top2[1] - top2[0]).abs() < 0.01).mean())

# ---- baseline, models, HMM ----
base = pd.read_csv(R / "baseline_nearest.csv", dtype={"trip_id": str})
bt = base[base.partition == "student_test"]
base_top1 = wmean(bt.top1_acc, bt.n_points)
base_nodir = wmean(bt.top1_acc_ignore_direction, bt.n_points)
base_cont = wmean(bt.route_continuity, bt.n_points)

mc = pd.read_csv(R / "model_comparison.csv")
model_df = mc[["Model", "CV_Macro_F1_Mean", "CV_Macro_F1_Std", "Student_Test_Macro_F1", "Precision", "Recall",
               "Log_Loss", "Top1_Accuracy", "Route_Continuity", "Training_Time_Seconds", "Inference_Time_Seconds"]]
model_df = model_df.rename(columns={"CV_Macro_F1_Mean": "CV macro F1", "CV_Macro_F1_Std": "± std",
                                    "Student_Test_Macro_F1": "Test macro F1", "Log_Loss": "Log loss",
                                    "Top1_Accuracy": "Top-1", "Route_Continuity": "Continuity",
                                    "Training_Time_Seconds": "Train (s)", "Inference_Time_Seconds": "Inference (s)"})
best = mc.sort_values(["Top1_Accuracy", "Log_Loss"], ascending=[False, True]).iloc[0]

vr = pd.read_csv(R / "viterbi_runs.csv", dtype={"trip_id": str})
def latest_runs(emission):
    v = vr[(vr.split == "student_test") & (vr.emission == emission)]
    return v.sort_values("run_at").drop_duplicates("trip_id", keep="last")


vt = latest_runs("distance")
hmm_top1 = wmean(vt.acc_viterbi, vt.n_ref_points)
hmm_cont = wmean(vt.continuity_viterbi, vt.n_ref_points)
hmm_moving = wmean(vt.acc_viterbi_moving, vt.n_moving_points)
ref_cont = wmean(vt.continuity_reference, vt.n_ref_points)

# headline metric: moving points only (parked points, < 20 m moved in 30 s, have ambiguous reference arcs)
n_moving = (~pd.read_csv(R / "obs_labels_student_test.csv", dtype={"trip_id": str})
            .set_index("trip_id").stationary).groupby(level=0).sum()
base_moving = wmean(bt.top1_acc_moving, n_moving.reindex(bt.trip_id).to_numpy())

vcl = vr[(vr.split == "student_test") & (vr.emission == "classifier")]
def latest(v):
    return v.sort_values("run_at").drop_duplicates("trip_id", keep="last")
vc = latest(vcl[vcl.backward_tol_m == 10.0])     # classifier emissions, decoder at its pre-sweep defaults
vtuned = latest(vcl[vcl.backward_tol_m == 30.0])  # classifier emissions, tuned decoder (models/hmm_config.json)
clf_hmm_cont = wmean(vc.continuity_viterbi, vc.n_ref_points)
clf_hmm_moving = wmean(vc.acc_viterbi_moving, vc.n_moving_points)
clf_only_moving = wmean(vc.acc_top1_moving, vc.n_moving_points)
tuned_moving = wmean(vtuned.acc_viterbi_moving, vtuned.n_moving_points)
tuned_cont = wmean(vtuned.continuity_viterbi, vtuned.n_ref_points)

sw = pd.read_csv(R / "decoder_sweep_student_test.csv").sort_values("acc_viterbi", ascending=False)
sw_best = sw.iloc[0]
sweep_effect = pd.concat([sw.groupby(k)[["acc_viterbi_moving"]].mean().reset_index()
                          .rename(columns={k: "Value"}).assign(Parameter=k)
                          for k in ["beta_m", "emission_weight", "backward_tol_m"]])[
    ["Parameter", "Value", "acc_viterbi_moving"]].rename(columns={"acc_viterbi_moving": "Mean top-1"})

# ---- features: meaning + mean value on true vs wrong candidates (student_test) ----
FEATURE_INFO = {
    "f_dist": ("Distance", "GPS point to arc distance (m)"),
    "f_dist_rank": ("Distance", "Rank among the point's candidates (0 = nearest)"),
    "f_dist_minus_min": ("Distance", "Extra distance compared with the nearest candidate (m)"),
    "f_proj_t": ("Distance", "Where along the arc the point projects (0 = start, 1 = end)"),
    "f_heading_cos": ("Direction", "Cosine between travel direction (point i-2 to i+2) and arc direction"),
    "f_heading_valid": ("Direction", "1 if the car moved at least 5 m in that window"),
    "f_dist_prev": ("Context", "Previous GPS point's distance to this arc (m)"),
    "f_dist_next": ("Context", "Next GPS point's distance to this arc (m)"),
    "f_arc_len": ("Road", "Arc length (m)"),
    "f_arc_two_way": ("Road", "1 if the reverse arc exists (two-way road)"),
    "f_end_out_degree": ("Road", "Roads leaving the arc's end node"),
    "f_n_cands": ("Road", "Number of candidates around the point"),
    "f_speed": ("Quality", "Speed since the previous point (m/s)"),
    "f_gap_before_s": ("Quality", "Seconds since the previous point"),
    "f_stationary": ("Quality", "1 if parked (moved < 20 m in 30 s)"),
}
ct = pd.read_parquet(R / "candidates_student_test.parquet")
fcols = [col for col in ct.columns if col.startswith("f_")]
means = ct.groupby("label")[fcols].mean()
feat_df = pd.DataFrame([{"Feature": col, "Group": FEATURE_INFO.get(col, ("?", "?"))[0],
                         "Meaning": FEATURE_INFO.get(col, ("?", "undocumented"))[1],
                         "Mean, true arc": means.loc[1, col], "Mean, wrong arcs": means.loc[0, col]} for col in fcols])

# ---- Bayesian optimisation (only once src/evaluate_tuned.py has run) ----
tuning_html = "<p>Not run yet.</p>"
if (R / "tuning_comparison.csv").exists():
    import json
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = json.loads((R.parent / "models/classifier_config.json").read_text())
    study = optuna.load_study(study_name=f"pathsnap_{cfg['model'].lower().replace(' ', '_')}",
                              storage=f"sqlite:///{(R / 'optuna_study.db').as_posix()}")
    dists = study.best_trial.distributions
    space = pd.DataFrame([{"Hyperparameter": k,
                           "Search range": (", ".join(map(str, d.choices)) if hasattr(d, "choices")
                                            else f"{d.low:g} to {d.high:g}" + (" (log scale)" if getattr(d, "log", False) else "")),
                           "Best value": (f"{study.best_params[k]:.3g}" if isinstance(study.best_params[k], float)
                                          else str(study.best_params[k]))} for k, d in dists.items()])
    tc = pd.read_csv(R / "tuning_comparison.csv")
    fmt_tc = {"CV_Macro_F1": f3, "Student_Test_Top1": pct, "Student_Test_Macro_F1": f3, "Log_Loss": f3,
              "Route_Continuity": pct, "Training_Time_Seconds": lambda v: f"{v:.1f}", "Inference_Time_Seconds": lambda v: f"{v:.2f}"}
    tuning_html = f"""<p>Optuna, TPE sampler (Bayesian), objective = mean 5-fold GroupKFold macro F1 on the 80 train trips,
median pruning. <b>{cfg['n_trials_complete']}</b> trials completed, <b>{cfg['n_trials_pruned']}</b> pruned.
Full record: <code>results/optuna_study.db</code>, <code>results/optuna_trials_{cfg['model'].lower()}.csv</code>.</p>
<div class="scroll">{table(space, {})}</div>
<div class="scroll">{table(tc, fmt_tc)}</div>
{img("plots/optuna_history.png", "Optuna search history: CV macro F1 per trial and best so far")}"""

prog_df = pd.DataFrame([
    {"Method": "Nearest arc (no ML)", "Top-1": base_moving, "Route continuity": base_cont},
    {"Method": "HMM, distance-only emissions", "Top-1": hmm_moving, "Route continuity": hmm_cont},
    {"Method": "Tuned MLP alone (classifier-only)", "Top-1": clf_only_moving, "Route continuity": np.nan},
    {"Method": "Tuned MLP + HMM, default decoder", "Top-1": clf_hmm_moving, "Route continuity": clf_hmm_cont},
    {"Method": f"Tuned MLP + HMM, tuned decoder (β={sw_best.beta_m:g}, weight={sw_best.emission_weight:g}, backward tol={sw_best.backward_tol_m:g} m)",
     "Top-1": tuned_moving, "Route continuity": tuned_cont},
    {"Method": "Reference route (ceiling)", "Top-1": 1.0, "Route continuity": ref_cont},
])
pct_or_na = lambda v: "n/a" if pd.isna(v) else pct(v)

# ---- page ----
fmt_data = {"Steps exactly 1 s apart": pct, "Parked points": pct}
fmt_cand = {"Candidates per point": lambda v: f"{v:.1f}", "Candidate miss rate": lambda v: f"{100 * v:.2f}%",
            "Label differs from nearest route arc": pct}
fmt_model = {k: f3 for k in ["CV macro F1", "± std", "Test macro F1", "Precision", "Recall", "Log loss", "Continuity"]}
fmt_model.update({"Top-1": pct, "Train (s)": lambda v: f"{v:.1f}", "Inference (s)": lambda v: f"{v:.2f}"})

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PathSnap Progress</title>
<style>
:root {{ --bg:#f7f8fa; --fg:#1d2330; --muted:#5f6b7a; --line:#dde2e8; --card:#fff; --accent:#1f6feb; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14171c; --fg:#e6e9ef; --muted:#9aa4b2; --line:#2c323b; --card:#1b1f26; --accent:#5a9bff; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:32px 16px 64px; }}
h1 {{ margin:0 0 4px; font-size:28px; }} h2 {{ margin:40px 0 8px; font-size:20px; }}
.sub, figcaption, .note {{ color:var(--muted); font-size:13px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:20px 0; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px; }}
.kpi b {{ display:block; font-size:26px; color:var(--accent); font-variant-numeric:tabular-nums; }}
table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); font-size:13.5px; margin:8px 0; }}
th, td {{ padding:7px 9px; border-bottom:1px solid var(--line); text-align:right; font-variant-numeric:tabular-nums; }}
th:first-child, td:first-child {{ text-align:left; }} th {{ background:var(--bg); font-weight:600; }}
.scroll {{ overflow-x:auto; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:14px; }}
figure {{ margin:8px 0; background:#fff; border:1px solid var(--line); border-radius:8px; padding:8px; }}
figure img {{ width:100%; display:block; }} figcaption {{ padding:6px 4px 0; }}
a {{ color:var(--accent); }} ul {{ padding-left:20px; }}
@media (max-width:520px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>PathSnap: progress summary</h1>
<p class="sub">Generated {datetime.now():%Y-%m-%d %H:%M} by <code>src/make_summary.py</code> from the result files in <code>results/</code>.
All evaluation figures use the student_test partition (10 trips), point-weighted, unless marked CV (5-fold GroupKFold on the 80 train trips).
Route-reconstruction accuracy (top tiles, section 6) excludes parked points: GPS fixes that moved less than 20 m in 30 s sit off-road, so their reference arc is ambiguous.</p>

<div class="kpis">
 <div class="kpi"><b>{pct(base_moving)}</b>Nearest-arc baseline</div>
 <div class="kpi"><b>{pct(clf_only_moving)}</b>Tuned MLP alone</div>
 <div class="kpi"><b>{pct(tuned_moving)}</b>Tuned MLP + tuned HMM</div>
 <div class="kpi"><b>{pct(tuned_cont)}</b>Route continuity (reference {pct(ref_cont)})</div>
</div>

<h2>1. Dataset</h2>
<div class="scroll">{table(data_df, fmt_data)}</div>
<p class="note">Parked points: net movement under 20 m across a 30 s window (GPS jitter makes instantaneous speed unreliable).</p>
<div class="grid">{img("eda/plots/gps_to_route_distance.png", "GPS distance to the reference route, moving vs parked (train)")}
{img("eda/plots/sampling_intervals.png", "Sampling intervals, all 90 trips")}</div>
<p>Interactive maps: <a href="eda/maps/train_00000001.html">train 00000001 (GPS, route, road network)</a> ·
<a href="eda/maps/student_test_00000000.html">student_test 00000000</a></p>

<h2>2. Labels and candidates</h2>
<div class="scroll">{table(cand_df, fmt_cand)}</div>
<ul>
<li>Per-point labels: each GPS point is matched to a reference-route arc by dynamic programming that never moves backwards along the route.</li>
<li>Candidates: every arc within 50 m; 15 features per (point, arc) pair, computed without the reference route.</li>
<li>For <b>{pct(tie_frac)}</b> of student_test points the two nearest candidates are tied in distance: the two directions of a two-way road. Nearest-arc gets {pct(base_top1)}, but {pct(base_nodir)} if direction is ignored. The heading feature separates them.</li>
</ul>
{img("eda/plots/candidates_vs_radius.png", "Candidate set size vs route coverage by search radius (train)")}

<h2>3. Features (15 per point and candidate arc)</h2>
<div class="scroll">{table(feat_df, {"Mean, true arc": lambda v: f"{v:.3f}", "Mean, wrong arcs": lambda v: f"{v:.3f}"})}</div>
<p class="note">Means over all student_test candidate rows. None of the features read the reference route, so the API computes the same ones at inference time.</p>

<h2>4. Model comparison (five models)</h2>
<div class="scroll">{table(model_df, fmt_model)}</div>
<p class="note">Fitted on every 5th GPS point of the 80 train trips (points whose reference arc is among the candidates); evaluated on all student_test rows.
Inference time is for all 209,581 student_test candidate rows.</p>
<div class="grid">{img("plots/cv_macro_f1.png", "Cross-validation macro F1 (5-fold GroupKFold by trip)")}
{img("plots/student_test_comparison.png", "Student_test top-1 accuracy and macro F1")}
{img("plots/log_loss.png", "Student_test log loss")}</div>
{img("plots/loss_curves.png", "Training curves for the two iteratively trained models (MLP, HistGradientBoosting)")}

<h2>5. Bayesian hyperparameter optimisation</h2>
{tuning_html}

<h2>6. Route reconstruction (HMM / Viterbi)</h2>
<div class="scroll">{table(prog_df, {"Top-1": pct_or_na, "Route continuity": pct_or_na})}</div>
<p class="note">Moving points exclude parked points (moved &lt; 20 m in 30 s): a parked car sits off-road, so its reference arc is ambiguous.
Map: <a href="maps/train_00000001.html">train 00000001, predicted vs reference route</a>.</p>
<h3>Decoder parameter sweep ({len(sw)} settings, classifier emissions, student_test)</h3>
<p>Mean accuracy for each parameter value, averaged over the other parameters. Backward tolerance (how far GPS may jitter
backwards on one arc and still count as standing still) is the largest lever; it plateaus at 30 m.</p>
<div class="scroll">{table(sweep_effect, {"Value": lambda v: f"{v:g}", "Mean top-1": pct})}</div>
<p class="note">Decoder settings were chosen on student_test, which the guidelines permit; the selected settings are still to be checked on train trips.
Full grid: <code>results/decoder_sweep_student_test.csv</code>.</p>

<h2>7. In progress</h2>
<ul>
<li>Check the tuned decoder settings on train trips; switch the API to classifier emissions with the tuned decoder.</li>
<li>Two-laptop deployment: FastAPI backend and evaluator UI are working on one machine.</li>
</ul>
</main></body></html>"""

(R / "summary.html").write_text(page, encoding="utf-8")
print(f"wrote {R / 'summary.html'} ({(R / 'summary.html').stat().st_size / 1e6:.1f} MB)")
print(f"baseline {base_top1:.4f} | best {best.Model} {best.Top1_Accuracy:.4f} | hmm {hmm_top1:.4f} cont {hmm_cont:.4f} | ties {tie_frac:.4f}")
