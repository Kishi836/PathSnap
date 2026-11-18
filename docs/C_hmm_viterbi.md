# Role C — HMM / Viterbi Route Reconstruction (Vamsi)

Read [WORK_SPLIT.md](WORK_SPLIT.md) first for the project overview, data format, partition rules and the candidates table.

## Your job in one line

The classifier scores each GPS point independently, so consecutive points can land on unconnected roads. You build an HMM + Viterbi decoder that picks one candidate per point so the sequence forms a connected, plausible route.

## Input

- Candidates per GPS point (`results/candidates_*.parquet` from A): `trip_id, obs_idx, arc_id, label, f_*`.
- Classifier probabilities from B (`models/classifier.joblib`). Until it exists, use a distance-only emission (below).
- Road network per trip: `.nodes` (lon, lat; row = node_id), `.arcs` (from, to; row = arc_id).

## Tasks

1. **Network helpers** (`src/network.py`): load a trip's graph; `connected(a, b)` = same arc, or `arcs[a].to == arcs[b].from`; shortest-path distance between arcs (Dijkstra on node graph, edge weight = arc length in metres — use haversine or a projected CRS, not degrees). Route continuity metric lives here (B uses it too).
2. **HMM definition** — write these down, the report asks for each:
   - Hidden states: the candidate arcs at each GPS point.
   - Emission P(GPS point | arc): classifier probability. Baseline: Gaussian on point-to-arc distance, `exp(-d² / 2σ²)`.
   - Transition P(arc_t+1 | arc_t): based on network connectivity, e.g. penalise |route distance − straight-line distance| between the two GPS points with an exponential (Newson & Krumm, 2009, "Hidden Markov Map Matching Through Noise and Sparseness", ACM SIGSPATIAL GIS). Unreachable → very small probability, not 0, so decoding never dead-ends.
3. **Viterbi** in log-space (`src/viterbi.py`): `decode(candidates_for_trip, emission_scores, graph) -> list of arc_id, one per GPS point`. Keep the candidate ↔ arc_id mapping exact.
4. **Edge cases** — the report asks how you handle each:
   - point with no candidates (skip it / carry previous state)
   - large time gaps in `.track` (check A's EDA for how big) — e.g. break into segments or loosen the transition
   - no connected path between consecutive candidates
5. **Experiments** on train / student_test only:
   - Classifier-only (top-1 per point) vs classifier + Viterbi: observation-level accuracy, route continuity, sequence-level score (e.g. fraction of trips with ≥ X% correct, or overlap of predicted vs reference arc sets).
   - Tune decoder params (σ, transition β) on train/student_test only. Log to `results/decoder_results.csv`.
   - Find and show cases where HMM did **not** help.
6. **Folium maps** (`src/maps.py`): layers for raw GPS points, nearby road network, reference route, predicted route; distinct colours, legend, `LayerControl`, sensible zoom. Save HTML to `results/maps/`. D can reuse the drawing code in the UI (minus the reference route).
7. **Export** decoder params to `models/hmm_config.json` for D.

## Report sections you draft

§5 (answers 1–6 + classifier-only vs HMM table) and §6 (Folium maps: screenshot in PDF, HTML + code in repo).

## Don'ts

- No evaluator data. Decoder params tuned on train/student_test only.
- Don't use the reference route inside `decode` — only for scoring afterwards.

## Viva prep

Be ready to explain: the Viterbi recurrence (δ_t(j) = max_i δ_{t-1}(i) + log A_ij + log B_j(o_t), with back-pointers), why log-space, how emissions/transitions were chosen, and what happens on gaps or disconnected candidates.
