# Role B — Models & Bayesian Tuning (Jose)

Read [WORK_SPLIT.md](WORK_SPLIT.md) first for the project overview, data format, partition rules and the candidates table you consume.

## Your job in one line

Train ≥ 5 classifiers that score "is this candidate arc the true arc for this GPS point?", compare them honestly, pick one, tune it with Bayesian optimisation, and save it for C (decoder) and D (API).

## Input

`results/candidates_train.parquet` and `results/candidates_student_test.parquet` (from A). One row per (GPS point, candidate arc). Target = `label`, features = all `f_*` columns. Group = `trip_id`.

Until A pushes, build the harness against that schema with a tiny fake DataFrame.

## Tasks

1. **Harness** (`src/train_models.py` or a notebook): load parquet → `GroupKFold` by `trip_id` on train (5 folds) → fit on train → evaluate on student_test. Same preprocessing for every model, fitted on train only (e.g. scaler inside a `Pipeline`).
2. **≥ 5 models from different families**, e.g. Logistic Regression, KNN, Random Forest / Extra Trees, HistGradientBoosting or LightGBM/XGBoost, MLP, SVM (on a subsample — SVM is slow on large data). The candidate set is imbalanced (few positives per point) — consider `class_weight`.
3. **Metrics per model** (log everything to `results/model_comparison.csv`):
   - CV Macro F1 (mean ± std over folds)
   - Student-test: Macro F1, Precision, Recall, Log Loss
   - **Observation-level Top-1 Accuracy**: per (trip_id, obs_idx), take the candidate with the highest predicted probability; correct if it equals `ref_arc_id`. Count candidate-miss points as wrong (and report the miss rate separately).
   - Training time, inference time
   - Route continuity (fraction of consecutive top-1 arcs that are the same arc or connected) — coordinate with C, who writes the connectivity helper.
4. **Plots** (`results/plots/`): CV Macro F1 bar chart across models; student-test Top-1 + Macro F1 comparison; Log Loss comparison; confusion matrices (optional); training/validation loss curves **only** for models that really have epochs (MLP `loss_curve_`, boosting eval history). Never draw curves for models that don't train iteratively.
5. **Pick the model.** Primary metric suggestion: student-test Top-1 accuracy (it is what the decoder and demo care about), with Log Loss as tie-breaker since C uses the probabilities as emissions. Write down why.
6. **Bayesian HPO** with Optuna (TPE sampler = Bayesian) on the chosen model:
   - objective = mean GroupKFold Macro F1 (or Top-1) on **train only**
   - define + justify ranges; fixed seed; ~30–50 trials
   - save the study (`results/optuna_study.db` or `results/optuna_trials.csv`)
   - refit best params on all of train → evaluate on student_test → table of untuned vs tuned (CV Macro F1, test Top-1, test Macro F1, Log Loss, inference time)
7. **Export** `models/classifier.joblib` + `models/classifier_config.json` (feature column order, params, seed). C and D load this.

## Report sections you draft

§3 (comparison table + one subsection per model answering: why suitable, strengths/limits, key hyperparameters + initial values, loss/objective, student-test results, over/underfitting evidence) and §4 (Bayesian HPO answers 1–6). Every number must come from `results/model_comparison.csv` or the Optuna record.

## Don'ts

- No evaluator data anywhere. Student_test is for comparison, not training.
- Don't pick by training accuracy.
- Don't split a trip across folds (that's why GroupKFold).
- Record random seeds and configs for every run.

## Viva prep

Be ready to explain: why GroupKFold, why each model was or wasn't suitable, what Macro F1 vs Top-1 measure differently, how TPE differs from grid/random search, and what the tuning gained.
