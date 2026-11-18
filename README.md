# PathSnap: Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University.

Noisy GPS tracks are matched to the road network: for every GPS point we predict which road segment (arc) the vehicle was on. Pipeline:

1. **Candidates**: road arcs within 50 m of each GPS point, with features.
2. **Classifier**: scores each (GPS point, candidate arc) pair.
3. **HMM / Viterbi**: picks one candidate per point so the route is connected.
4. **API + UI**: FastAPI backend (laptop 1) runs the pipeline for a selected trip; the UI (laptop 2) shows the predicted route and accuracy.

## Repository layout

| Path | Contents |
|---|---|
| `Student folder/Student folder/` | Raw data: `train/` (80 trips), `student_test/` (10 trips) |
| `PathSnap_student_materials/` | Hackathon guideline PDFs |
| `src/data.py` | Data loading, metric projection, point-to-arc distance, candidate search, parked-point detection |
| `src/eda.py` | Exploratory data analysis |
| `src/build_candidates.py` | Per-point reference labels, candidate generation, features |
| `src/baseline_nearest.py` | No-ML baseline: nearest candidate arc |
| `src/train_models.py` | Five classifiers, GroupKFold CV, student_test metrics and comparison plots |
| `src/tune_models.py` | Bayesian hyperparameter optimisation (Optuna TPE) |
| `src/plot_loss_curves.py` | Epoch-wise training curves (MLP, HistGradientBoosting) |
| `src/network.py`, `src/viterbi.py`, `src/run_viterbi.py` | Road graph, HMM / Viterbi decoder, decoder evaluation |
| `src/make_summary.py` | Builds `results/summary.html` from the result files |
| `api/main.py` | FastAPI backend (laptop 1) |
| `ui/index.html` | Evaluator UI (laptop 2), opens in any browser |
| `models/` | Decoder configuration |
| `results/eda/` | EDA plots, Folium maps, per-trip inventory, summary |
| `results/` | Candidate tables, labels, experiment results |

## Data format

Each trip folder `<trip_id>/` has four whitespace-separated files without headers:

| File | Columns |
|---|---|
| `.track` | longitude, latitude, timestamp (s) |
| `.nodes` | longitude, latitude (row index = node id) |
| `.arcs` | from node, to node (row index = arc id) |
| `.route` | arc ids of the reference route, in driving order |

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

## Running

All commands run from the repository root.

```bash
python src/data.py               # self-checks for the geometry helpers
python src/eda.py                # EDA -> results/eda/   (~6 min)
python src/build_candidates.py   # labels + candidates -> results/   (~5 min)
python src/baseline_nearest.py   # nearest-arc baseline -> results/baseline_nearest.csv
python src/train_models.py       # five-model comparison -> results/model_comparison.csv, results/plots/   (~10 min)
python src/plot_loss_curves.py   # training curves -> results/plots/loss_curves.png
python -m src.run_viterbi --split student_test --trips all --no-map   # HMM decoder evaluation
python src/make_summary.py       # progress page -> results/summary.html
```

## Deployment (two laptops)

Laptop 1 (API), from the repository root:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Optional settings: `PATHSNAP_DATA_ROOT` (folder holding the trip partitions) and `PATHSNAP_SPLIT` (partition shown in the UI, default `student_test`). Allow port 8000 through the firewall.

Laptop 2 (UI): open `ui/index.html` in a browser, enter `http://<laptop-1-IP>:8000`, click **Connect**, pick a trip, click **Run prediction**. The API prints one log line per request. The reference route is used only inside the API to compute accuracy and is never sent to the UI.

## Outputs used by the rest of the pipeline

| File | One row per | Key columns |
|---|---|---|
| `results/obs_labels_{train,student_test}.csv` | GPS point | `trip_id, obs_idx, ref_arc_id, ref_dist_m, stationary` |
| `results/candidates_{train,student_test}.parquet` | (GPS point, candidate arc) | `trip_id, obs_idx, arc_id, label, f_*` features |

`trip_features()` in `src/build_candidates.py` builds candidates and features from the GPS track and road network only; it never reads the reference route, so the same function is used at inference time.

## Data partitions

| Partition | Used for |
|---|---|
| train (80 trips) | Model fitting, cross-validation, hyperparameter search |
| student_test (10 trips) | Model comparison, decoder tuning |
| evaluator (10 trips, held by organisers) | Final demo only |

## Team

| Role | Member |
|---|---|
| Data, EDA, features | Kshitij |
| Models, Bayesian optimisation | Jose |
| HMM / Viterbi | Vamsi |
| API, UI, deployment | Mahit |
