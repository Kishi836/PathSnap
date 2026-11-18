"""Refit the best Optuna configuration, compare untuned vs tuned on the same data, save the final model.

Run from the repo root after src/tune_models.py:  python src/evaluate_tuned.py [--model MLP]
Writes: results/tuning_comparison.csv, results/optuna_trials_<model>.csv, results/plots/optuna_history.png,
        models/classifier.joblib, models/classifier_config.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from sklearn.metrics import f1_score, log_loss

sys.path.insert(0, str(Path(__file__).parent))
from train_models import (COMPARISON_PATH, PLOTS_DIR, RANDOM_SEED, RESULTS_DIR, ROOT_DIR, TEST_LABELS_PATH, TEST_PATH,
                          TRAIN_PATH, calculate_top1, create_models, get_positive_probabilities, mean_route_continuity,
                          prepare_data, training_subset, validate_data)
from tune_models import STUDY_DB


def pipeline_params(best_params):
    """Optuna parameter names -> pipeline parameters (layer shapes are stored as strings like "64-32")."""
    out = {}
    for k, v in best_params.items():
        if k == "hidden_layer_sizes":
            v = tuple(int(n) for n in str(v).split("-"))
        out[f"classifier__{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MLP")
    model_name = ap.parse_args().model
    safe = model_name.lower().replace(" ", "_")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=f"pathsnap_{safe}", storage=f"sqlite:///{STUDY_DB.resolve().as_posix()}")
    trials = study.trials_dataframe()
    trials.to_csv(RESULTS_DIR / f"optuna_trials_{safe}.csv", index=False)
    complete = trials[trials.state == "COMPLETE"].sort_values("number")

    # search record plot: every completed trial and the best value found so far
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(complete.number, complete.value, s=18, label="completed trial")
    ax.step(complete.number, complete.value.cummax(), where="post", c="tab:red", label="best so far")
    pruned = trials[trials.state == "PRUNED"]
    if len(pruned):
        ax.scatter(pruned.number, [complete.value.min()] * len(pruned), marker="x", c="0.6", label="pruned (stopped early)")
    ax.set(xlabel="trial", ylabel="CV macro F1 (5-fold GroupKFold)", title=f"Optuna TPE search: {model_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "optuna_history.png", dpi=130)
    plt.close(fig)

    # refit best configuration on the same training subset as the untuned model
    train = training_subset(pd.read_parquet(TRAIN_PATH))
    test = pd.read_parquet(TEST_PATH)
    labels = pd.read_csv(TEST_LABELS_PATH, dtype={"trip_id": str})
    features = validate_data(train, "Training data")
    X, y, _ = prepare_data(train, features)
    X_test, y_test, _ = prepare_data(test, features)

    model = create_models()[model_name].set_params(**pipeline_params(study.best_params))
    t0 = time.perf_counter()
    model.fit(X, y)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X_test)
    prob = get_positive_probabilities(model, X_test)
    infer_s = time.perf_counter() - t0
    top1, _, _, _, best = calculate_top1(test, prob, labels)

    untuned = pd.read_csv(COMPARISON_PATH).set_index("Model").loc[model_name]
    comparison = pd.DataFrame([
        {"Model": f"{model_name} (untuned)", "CV_Macro_F1": untuned.CV_Macro_F1_Mean,
         "Student_Test_Top1": untuned.Top1_Accuracy, "Student_Test_Macro_F1": untuned.Student_Test_Macro_F1,
         "Log_Loss": untuned.Log_Loss, "Route_Continuity": untuned.Route_Continuity,
         "Training_Time_Seconds": untuned.Training_Time_Seconds, "Inference_Time_Seconds": untuned.Inference_Time_Seconds},
        {"Model": f"{model_name} (tuned)", "CV_Macro_F1": study.best_value,
         "Student_Test_Top1": top1, "Student_Test_Macro_F1": f1_score(y_test, pred, average="macro"),
         "Log_Loss": log_loss(y_test, prob, labels=[0, 1]), "Route_Continuity": mean_route_continuity(best),
         "Training_Time_Seconds": train_s, "Inference_Time_Seconds": infer_s},
    ])
    comparison.to_csv(RESULTS_DIR / "tuning_comparison.csv", index=False)

    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "classifier.joblib")
    (models_dir / "classifier_config.json").write_text(json.dumps({
        "model": model_name, "feature_columns": features, "best_params": study.best_params,
        "best_cv_macro_f1": study.best_value, "n_trials_complete": int(len(complete)),
        "n_trials_pruned": int(len(pruned)), "random_seed": RANDOM_SEED,
        "training_subset": "every 5th GPS point of the 80 train trips, points whose reference arc is a candidate",
    }, indent=2))

    print(comparison.round(4).to_string(index=False))
    print(f"trials: {len(complete)} complete, {len(pruned)} pruned | best params {study.best_params}")


if __name__ == "__main__":
    main()
