# PathSnap — Work Split

| Role | Owner | GitHub | Brief |
|---|---|---|---|
| A — Data & features | Kshitij | Kishi836 | (this file + EDA code) |
| B — Models & Bayesian tuning | Jose | jjosetanmay06 | [B_models.md](B_models.md) |
| C — HMM / Viterbi | Vamsi | NagaVamsi-1 | [C_hmm_viterbi.md](C_hmm_viterbi.md) |
| D — API, UI, deployment | Mahit | Kumar-Mahit | [D_api_ui.md](D_api_ui.md) |

Every member must be able to explain **every** part in the viva, not just their own. Read all four briefs.

## The project in one paragraph

GPS points from a car are noisy: they drift onto parallel streets or jump across buildings. Given a trip's noisy GPS track and its road network, we must recover which road segment (arc) the car was actually on at every GPS point. Pipeline:

1. **Candidates** (A): for each GPS point, pick nearby arcs as candidates.
2. **Classifier** (B): score each (GPS point, candidate arc) pair: probability this candidate is the true arc.
3. **HMM / Viterbi** (C): pick one candidate per point so the whole sequence is a connected, plausible route.
4. **API + UI** (D): evaluator picks a trip in the UI (laptop 2), API (laptop 1) runs 1→3 and returns predicted route + accuracy.

## Data (verified on disk)

`Student folder/Student folder/{train,student_test}/<trip_id>/<trip_id>.<ext>` — 80 train trips, 10 student_test trips. The 10 evaluator trips are **not** given to us.

| File | Columns (whitespace-separated, no header) |
|---|---|
| `.track` | longitude, latitude, timestamp (s) — ~1 Hz GPS |
| `.nodes` | longitude, latitude — row index = node_id (0-based) |
| `.arcs` | from_node_id, to_node_id — row index = arc_id (0-based) |
| `.route` | arc_id — true arcs in driving order (**fewer rows than `.track`**) |

Each trip has its **own** road network (trip 00000001: 88,809 nodes, 165,790 arcs). Coordinates are lon/lat in degrees — never use plain Euclidean distance on them.

## Partitions — hard rules

| Partition | Use |
|---|---|
| train (80) | fitting, cross-validation, Bayesian tuning |
| student_test (10) | comparing models, decoder tuning |
| evaluator (10, not ours) | final demo only — no training, tuning or choices based on it |

No hardcoded predictions. Ground-truth route stays server-side, never sent to the model or UI.

## Shared contracts (A produces, B/C/D consume)

**`results/candidates_{train,student_test}.parquet`** — one row per (GPS point, candidate arc):

| Column | Meaning |
|---|---|
| `trip_id` | e.g. `00000001` (string) |
| `obs_idx` | row index in `.track` |
| `arc_id` | candidate arc |
| `label` | 1 if `arc_id` is the reference arc for this point, else 0 |
| `f_*` | feature columns (distance, heading difference, …) — final list set by A |

**`results/obs_labels_{train,student_test}.csv`** — one row per GPS point: `trip_id, obs_idx, ref_arc_id` (empty if no reference arc could be assigned). A point whose `ref_arc_id` is not among its candidates is a **candidate miss**.

The column names above are fixed; feature columns may grow. A will post in the group when the first version is pushed.

## Proposed repo layout

```
Student folder/   raw data (untouched)
src/              python code (data, features, models, hmm)
api/              FastAPI backend
ui/               evaluator UI
models/           saved model + config artifacts
results/          CSVs, logs, Optuna study, plots, folium maps
docs/             these briefs, report drafts
```

## Git workflow

- Branch per task (`b/models`, `c/viterbi`, `d/api` …), push, open a PR into `main`.
- Kshitij reviews and merges. Don't push straight to `main`.
- Never commit secrets. Keep files < 50 MB (GitHub hard limit 100 MB).

## AI-use rules (from the Assessment & Plagiarism Policy)

- Allowed: AI-drafted code and report text that **you** then read, change and can defend.
- Prohibited: fabricated metrics, plots, logs or citations; describing experiments that were not run; sharing anything with other teams.
- Every number/plot in the report must be reproducible from code in this repo.

## Phase 1 (before lunch)

| Role | Target |
|---|---|
| A | EDA done; first candidates/labels parquet pushed |
| B | train/eval harness ready; 2–3 models trained once candidates land |
| C | Viterbi working on one trip with distance-only emissions |
| D | API + UI running end-to-end on two laptops with a dummy predictor |
