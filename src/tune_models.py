
from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import optuna

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from train_models import (
    ROOT_DIR,
    TRAIN_PATH,
    RANDOM_SEED,
    N_SPLITS,
    create_models,
    training_subset
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

RESULTS_DIR = ROOT_DIR / "results"
STUDY_DB = RESULTS_DIR / "optuna_study.db"
TRIALS_DIR = RESULTS_DIR / "optuna_trials"

TRIALS_DIR.mkdir(parents=True, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==================================================
# 2. HYPERPARAMETER SEARCH SPACES
# ==================================================

def suggest_parameters(trial, model_name):

    if model_name == "Logistic Regression":

        return {
            "classifier__C": trial.suggest_float(
                "C", 0.001, 100, log=True
            ),
            "classifier__class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", None]
            ),
            "classifier__solver": trial.suggest_categorical(
                "solver", ["lbfgs", "liblinear"]
            )
        }

    elif model_name == "KNN":

        return {
            "classifier__n_neighbors": trial.suggest_int(
                "n_neighbors", 3, 15
            ),
            "classifier__weights": trial.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),
            "classifier__p": trial.suggest_int(
                "p", 1, 2
            )
        }

    elif model_name == "Random Forest":

        return {
            "classifier__n_estimators": trial.suggest_int(
                "n_estimators", 100, 500, step=50
            ),
            "classifier__max_depth": trial.suggest_categorical(
                "max_depth", [None, 10, 20, 30, 40]
            ),
            "classifier__min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 20
            ),
            "classifier__min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 10
            ),
            "classifier__max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "classifier__class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", None]
            )
        }

    elif model_name == "HistGradientBoosting":

        return {
            "classifier__learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "classifier__max_iter": trial.suggest_int(
                "max_iter", 50, 300, step=25
            ),
            "classifier__max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes", 7, 63, step=4
            ),
            "classifier__l2_regularization": trial.suggest_float(
                "l2_regularization", 0.001, 10, log=True
            ),
            "classifier__min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 10, 40
            ),
            "classifier__class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", None]
            )
        }

    elif model_name == "MLP":

        # Optuna stores only plain values, so layer shapes are named as strings, e.g. "64-32"
        layers = trial.suggest_categorical(
            "hidden_layer_sizes", ["32", "64", "64-32", "128-64"]
        )

        return {
            "classifier__hidden_layer_sizes": tuple(int(n) for n in layers.split("-")),
            "classifier__activation": trial.suggest_categorical(
                "activation", ["relu", "tanh"]
            ),
            "classifier__alpha": trial.suggest_float(
                "alpha", 0.00001, 0.1, log=True
            ),
            "classifier__learning_rate_init": trial.suggest_float(
                "learning_rate_init", 0.0001, 0.01, log=True
            ),
            # ~292k rows per fold: batches below 128 mean thousands of updates per epoch and
            # several hours for the study, so the search covers 128-512
            "classifier__batch_size": trial.suggest_categorical(
                "batch_size", [128, 256, 512]
            )
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ==================================================
# 3. BAYESIAN OPTIMISATION OBJECTIVE
# ==================================================

def objective(trial, model_name, X, y, groups):

    base_model = create_models()[model_name]

    parameters = suggest_parameters(
        trial,
        model_name
    )

    model = clone(base_model)
    model.set_params(**parameters)

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        group_kfold.split(X, y, groups=groups),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        fold_model = clone(model)

        fold_model.fit(
            X_train,
            y_train
        )

        predictions = fold_model.predict(X_val)

        score = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )

        fold_scores.append(score)

        trial.report(
            float(np.mean(fold_scores)),
            step=fold
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


# ==================================================
# 4. LOAD TRAINING DATA
# ==================================================

def load_training_data():

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_PATH}"
        )

    # same fitting subset as train_models.py, so tuned and untuned runs are comparable
    train = training_subset(pd.read_parquet(TRAIN_PATH))

    feature_columns = [
        col for col in train.columns
        if col.startswith("f_")
    ]

    if not feature_columns:
        raise ValueError("No f_* features found.")

    if train["label"].nunique() != 2:
        raise ValueError(
            "Training data must contain both classes."
        )

    if train[feature_columns].isna().any().any():
        raise ValueError("Missing feature values found.")

    X = train[feature_columns].copy()
    y = train["label"].astype(int).copy()
    groups = train["trip_id"].copy()

    if groups.nunique() < N_SPLITS:
        raise ValueError(
            "At least five distinct trips are required."
        )

    return X, y, groups, feature_columns


# ==================================================
# 5. RUN OPTUNA STUDY
# ==================================================

def run_optimization(
    model_name,
    X,
    y,
    groups,
    n_trials,
    worker=0
):

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        model_name.lower()
    )

    study_name = f"pathsnap_{safe_name}"

    storage = (
        f"sqlite:///{STUDY_DB.resolve().as_posix()}"
    )

    # Several worker processes can share this SQLite study; each needs its own seed,
    # or they would all propose the same first trials.
    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_SEED + worker
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True
    )

    print("\nBAYESIAN OPTIMISATION")
    print("=" * 55)
    print(f"Selected model: {model_name}")
    print(f"Trials to run: {n_trials}")
    print(f"Validation: {N_SPLITS}-fold GroupKFold")
    print("Objective: Mean CV Macro F1")
    print("Sampler: TPE")

    study.optimize(
        lambda trial: objective(
            trial,
            model_name,
            X,
            y,
            groups
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("\nOPTIMISATION COMPLETED")
    print("-" * 55)

    print(f"Best CV Macro F1: {study.best_value:.5f}")
    print("Best parameters:")

    for parameter, value in study.best_params.items():
        print(f"{parameter}: {value}")

    return study


# ==================================================
# 6. SAVE OPTUNA RESULTS
# ==================================================

def save_results(study, model_name, feature_columns):

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        model_name.lower()
    )

    trials_path = TRIALS_DIR / f"{safe_name}_trials.csv"

    best_params_path = (
        RESULTS_DIR / f"{safe_name}_best_params.json"
    )

    study.trials_dataframe().to_csv(
        trials_path,
        index=False
    )

    best_results = {
        "model": model_name,
        "best_cv_macro_f1": study.best_value,
        "best_parameters": study.best_params,
        "feature_columns": feature_columns,
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS
    }

    with open(best_params_path, "w") as file:
        json.dump(
            best_results,
            file,
            indent=4
        )

    print("\nRESULTS SAVED")
    print(f"Study database: {STUDY_DB}")
    print(f"Trial history: {trials_path}")
    print(f"Best parameters: {best_params_path}")


# ==================================================
# 7. MAIN EXECUTION
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description="PathSnap Bayesian Hyperparameter Optimisation"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=list(create_models().keys()),
        help="Model to optimise."
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials."
    )

    parser.add_argument(
        "--worker",
        type=int,
        default=0,
        help="Worker number when several processes share the study (sets the sampler seed)."
    )

    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1.")

    X, y, groups, feature_columns = load_training_data()

    print("\nTRAINING DATA SUMMARY")
    print("-" * 50)
    print(f"Rows: {len(X)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Trips: {groups.nunique()}")

    study = run_optimization(
        args.model,
        X,
        y,
        groups,
        args.trials,
        args.worker
    )

    save_results(
        study,
        args.model,
        feature_columns
    )

    print("\nPhase 4 optimisation completed!")


if __name__ == "__main__":
    main()
