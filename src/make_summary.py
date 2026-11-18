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
vt = vr[vr.split == "student_test"].sort_values("run_at").drop_duplicates("trip_id", keep="last")
hmm_top1 = wmean(vt.acc_viterbi, vt.n_ref_points)
hmm_cont = wmean(vt.continuity_viterbi, vt.n_ref_points)
ref_cont = wmean(vt.continuity_reference, vt.n_ref_points)

prog_df = pd.DataFrame([
    {"Method": "Nearest arc (no ML)", "Top-1": base_top1, "Route continuity": base_cont},
    {"Method": f"Best classifier alone ({best.Model})", "Top-1": best.Top1_Accuracy, "Route continuity": best.Route_Continuity},
    {"Method": "HMM / Viterbi, distance-only emissions", "Top-1": hmm_top1, "Route continuity": hmm_cont},
    {"Method": "Reference route (ceiling)", "Top-1": 1.0, "Route continuity": ref_cont},
])

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
All evaluation figures use the student_test partition (10 trips), point-weighted, unless marked CV (5-fold GroupKFold on the 80 train trips).</p>

<div class="kpis">
 <div class="kpi"><b>{pct(base_top1)}</b>Nearest-arc baseline, top-1</div>
 <div class="kpi"><b>{pct(best.Top1_Accuracy)}</b>Best classifier ({html.escape(best.Model)}), top-1</div>
 <div class="kpi"><b>{pct(hmm_top1)}</b>HMM / Viterbi (distance-only), top-1</div>
 <div class="kpi"><b>{pct(hmm_cont)}</b>HMM route continuity (reference {pct(ref_cont)})</div>
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

<h2>3. Model comparison (five models)</h2>
<div class="scroll">{table(model_df, fmt_model)}</div>
<p class="note">Fitted on every 5th GPS point of the 80 train trips (points whose reference arc is among the candidates); evaluated on all student_test rows.
Inference time is for all 209,581 student_test candidate rows.</p>
<div class="grid">{img("plots/cv_macro_f1.png", "Cross-validation macro F1 (5-fold GroupKFold by trip)")}
{img("plots/student_test_comparison.png", "Student_test top-1 accuracy and macro F1")}
{img("plots/log_loss.png", "Student_test log loss")}</div>
{img("plots/loss_curves.png", "Training curves for the two iteratively trained models (MLP, HistGradientBoosting)")}

<h2>4. Route reconstruction (HMM / Viterbi)</h2>
<div class="scroll">{table(prog_df, {"Top-1": pct, "Route continuity": pct})}</div>
<p class="note">HMM run: distance-only emissions, σ and β estimated on train only. Map: <a href="maps/train_00000001.html">train 00000001, predicted vs reference route</a>.</p>

<h2>5. In progress</h2>
<ul>
<li>Bayesian hyperparameter optimisation (Optuna TPE) of the selected model.</li>
<li>Classifier probabilities as HMM emissions, and classifier-only vs classifier + HMM comparison.</li>
<li>Two-laptop deployment: FastAPI backend and evaluator UI are working on one machine.</li>
</ul>
</main></body></html>"""

(R / "summary.html").write_text(page, encoding="utf-8")
print(f"wrote {R / 'summary.html'} ({(R / 'summary.html').stat().st_size / 1e6:.1f} MB)")
print(f"baseline {base_top1:.4f} | best {best.Model} {best.Top1_Accuracy:.4f} | hmm {hmm_top1:.4f} cont {hmm_cont:.4f} | ties {tie_frac:.4f}")
