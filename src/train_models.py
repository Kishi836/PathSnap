
from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    log_loss
)


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

TRAIN_PATH = RESULTS_DIR / "candidates_train.parquet"
TEST_PATH = RESULTS_DIR / "candidates_student_test.parquet"

TRAIN_LABELS_PATH = RESULTS_DIR / "obs_labels_train.csv"
TEST_LABELS_PATH = RESULTS_DIR / "obs_labels_student_test.csv"

COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"

RANDOM_SEED = 42
N_SPLITS = 5


# ==================================================
# 2. LOAD ACTUAL DATA
# ==================================================

def load_data():

    required_files = [
        TRAIN_PATH,
        TEST_PATH,
        TRAIN_LABELS_PATH,
        TEST_LABELS_PATH
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    train_labels = pd.read_csv(
        TRAIN_LABELS_PATH,
        dtype={"trip_id": str}
    )

    test_labels = pd.read_csv(
        TEST_LABELS_PATH,
        dtype={"trip_id": str}
    )

    return train, test, train_labels, test_labels


# ==================================================
# 3. VALIDATE DATA
# ==================================================

def validate_data(df, dataset_name):

    required_columns = {
        "trip_id",
        "obs_idx",
        "arc_id",
        "label"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} missing columns: {missing}"
        )

    feature_columns = [
        col for col in df.columns
        if col.startswith("f_")
    ]

    if not feature_columns:
        raise ValueError(
            f"No f_* features found in {dataset_name}"
        )

    if df.empty:
        raise ValueError(f"{dataset_name} is empty.")

    if df["label"].isna().any():
        raise ValueError("Missing target labels.")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary: 0 or 1.")

    if df[feature_columns].isna().any().any():
        raise ValueError("Feature values contain missing data.")

    if not all(
        pd.api.types.is_numeric_dtype(df[col])
        for col in feature_columns
    ):
        raise ValueError("All features must be numeric.")

    if not np.isfinite(
        df[feature_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Features contain infinite values.")

    return feature_columns


def validate_reference_labels(labels, dataset_name):

    required = {"trip_id", "obs_idx", "ref_arc_id"}

    missing = required - set(labels.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} missing columns: {missing}"
        )

    if labels.duplicated(["trip_id", "obs_idx"]).any():
        raise ValueError(
            f"{dataset_name} contains duplicate observations."
        )

    labels = labels.copy()

    labels["ref_arc_id"] = pd.to_numeric(
        labels["ref_arc_id"],
        errors="coerce"
    )

    return labels


# ==================================================
# 4. PREPARE DATA
# ==================================================

def prepare_data(df, feature_columns):

    X = df[feature_columns].copy()
    y = df["label"].astype(int).copy()
    groups = df["trip_id"].copy()

    return X, y, groups


# ==================================================
# 5. GENERATE FAKE DATA
# ==================================================

def create_fake_data(number_of_trips=10):

    rng = np.random.default_rng(RANDOM_SEED)

    rows = []
    reference_rows = []

    for trip in range(number_of_trips):

        trip_id = f"{trip:08d}"

        for obs_idx in range(5):

            correct_arc = obs_idx + trip * 10

            reference_rows.append({
                "trip_id": trip_id,
                "obs_idx": obs_idx,
                "ref_arc_id": correct_arc
            })

            for candidate in range(3):

                label = int(candidate == 0)

                rows.append({
                    "trip_id": trip_id,
                    "obs_idx": obs_idx,
                    "arc_id": correct_arc + candidate * 100,
                    "label": label,

                    "f_distance": (
                        rng.uniform(0, 3)
                        if label else rng.uniform(5, 20)
                    ),

                    "f_heading_diff": (
                        rng.uniform(0, 15)
                        if label else rng.uniform(30, 180)
                    ),

                    "f_speed": rng.uniform(10, 60)
                })

    candidates = pd.DataFrame(rows)
    references = pd.DataFrame(reference_rows)

    return candidates, references


# ==================================================
# 6. GROUP K-FOLD VALIDATION
# ==================================================

def check_group_kfold(X, y, groups):

    if groups.nunique() < N_SPLITS:
        raise ValueError(
            f"At least {N_SPLITS} trips are required."
        )

    gkf = GroupKFold(n_splits=N_SPLITS)

    print("\nGROUP K-FOLD VALIDATION")
    print("-" * 50)

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X, y, groups=groups),
        start=1
    ):

        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])

        assert train_groups.isdisjoint(val_groups)

        print(
            f"Fold {fold}: "
            f"Train rows = {len(train_idx)}, "
            f"Validation rows = {len(val_idx)}, "
            f"Train trips = {len(train_groups)}, "
            f"Validation trips = {len(val_groups)}"
        )

    print("\nNo trip leakage detected.")


# ==================================================
# 7. DEFINE FIVE ML MODELS
# ==================================================

def create_models():

    return {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance"
            ))
        ]),

        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ))
        ]),

        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.1,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=RANDOM_SEED
            ))
        ]),

        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                random_state=RANDOM_SEED
            ))
        ])
    }


# ==================================================
# 8. GET POSITIVE CLASS PROBABILITIES
# ==================================================

def get_positive_probabilities(model, X):

    probabilities = model.predict_proba(X)

    classes = model.named_steps["classifier"].classes_

    positive_index = np.where(classes == 1)[0]

    if len(positive_index) != 1:
        raise ValueError(
            "Model did not learn both binary classes."
        )

    return probabilities[:, positive_index[0]]


# ==================================================
# 9. OBSERVATION-LEVEL TOP-1 ACCURACY
# ==================================================

def calculate_top1(candidates, probabilities, references):

    scored = candidates[
        ["trip_id", "obs_idx", "arc_id"]
    ].copy()

    scored["probability"] = probabilities

    # Select the highest-scoring candidate per GPS observation.
    best_candidates = (
        scored.sort_values(
            ["trip_id", "obs_idx", "probability"],
            ascending=[True, True, False]
        )
        .drop_duplicates(["trip_id", "obs_idx"])
    )

    # Only observations with an available reference arc
    # can be evaluated for Top-1 accuracy.
    valid_references = references.dropna(
        subset=["ref_arc_id"]
    ).copy()

    merged = best_candidates.merge(
        valid_references,
        on=["trip_id", "obs_idx"],
        how="right",
        indicator=True
    )

    # A reference arc missing from the candidate list
    # is counted as an incorrect prediction.
    candidate_misses = (
        merged["_merge"] == "right_only"
    )

    correct = (
        (merged["_merge"] == "both")
        & (
            merged["arc_id"]
            == merged["ref_arc_id"]
        )
    )

    total = len(merged)

    if total == 0:
        return np.nan, np.nan, 0, 0

    accuracy = correct.sum() / total

    miss_count = candidate_misses.sum()
    miss_rate = miss_count / total

    return (
        accuracy,
        miss_rate,
        int(correct.sum()),
        int(miss_count)
    )


# ==================================================
# 10. CROSS-VALIDATION MACRO F1
# ==================================================

def calculate_cv_score(model, X, y, groups):

    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_scores = []

    for train_idx, val_idx in gkf.split(
        X, y, groups=groups
    ):

        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]

        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]

        fold_model = clone(model)

        fold_model.fit(
            X_fold_train,
            y_fold_train
        )

        predictions = fold_model.predict(X_fold_val)

        score = f1_score(
            y_fold_val,
            predictions,
            average="macro",
            zero_division=0
        )

        fold_scores.append(score)

    return (
        float(np.mean(fold_scores)),
        float(np.std(fold_scores))
    )


# ==================================================
# 11. EVALUATE ALL FIVE MODELS
# ==================================================

def evaluate_models(
    X_train,
    y_train,
    groups,
    X_test,
    y_test,
    test_candidates,
    test_references
):

    models = create_models()

    results = []

    print("\nMODEL EVALUATION")
    print("=" * 65)

    for name, model in models.items():

        print(f"\nEvaluating: {name}")

        # Cross-validation
        cv_mean, cv_std = calculate_cv_score(
            model,
            X_train,
            y_train,
            groups
        )

        # Measure full training time
        start_time = time.perf_counter()

        model.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time

        # Measure inference time
        start_time = time.perf_counter()

        predictions = model.predict(X_test)
        probabilities = get_positive_probabilities(
            model, X_test
        )

        inference_time = time.perf_counter() - start_time

        # Student-test metrics
        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        loss = log_loss(
            y_test,
            probabilities,
            labels=[0, 1]
        )

        # Observation-level Top-1
        top1, miss_rate, correct_count, miss_count = (
            calculate_top1(
                test_candidates,
                probabilities,
                test_references
            )
        )

        result = {
            "Model": name,
            "CV_Macro_F1_Mean": cv_mean,
            "CV_Macro_F1_Std": cv_std,
            "Student_Test_Macro_F1": macro_f1,
            "Precision": precision,
            "Recall": recall,
            "Log_Loss": loss,
            "Top1_Accuracy": top1,
            "Candidate_Miss_Rate": miss_rate,
            "Top1_Correct_Count": correct_count,
            "Candidate_Miss_Count": miss_count,
            "Training_Time_Seconds": training_time,
            "Inference_Time_Seconds": inference_time,
            "Route_Continuity": np.nan
        }

        results.append(result)

        print(f"CV Macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"Test Macro F1: {macro_f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Log Loss: {loss:.4f}")
        print(f"Top-1 Accuracy: {top1:.4f}")
        print(f"Candidate Miss Rate: {miss_rate:.4f}")
        print(f"Training time: {training_time:.4f}s")
        print(f"Inference time: {inference_time:.4f}s")

    results_df = pd.DataFrame(results)

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    results_df.to_csv(
        COMPARISON_PATH,
        index=False
    )

    print("\nComparison saved to:")
    print(COMPARISON_PATH)

    return results_df


# ==================================================
# 12. GENERATE COMPARISON PLOTS
# ==================================================

def generate_plots(results_df):

    PLOTS_DIR.mkdir(exist_ok=True, parents=True)

    # Plot 1: CV Macro F1
    plt.figure(figsize=(10, 6))

    plt.bar(
        results_df["Model"],
        results_df["CV_Macro_F1_Mean"],
        yerr=results_df["CV_Macro_F1_Std"],
        capsize=5
    )

    plt.ylabel("Macro F1")
    plt.title("Cross-Validation Macro F1 Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "cv_macro_f1.png")
    plt.close()

    # Plot 2: Student-test Top-1 and Macro F1
    plt.figure(figsize=(10, 6))

    x = np.arange(len(results_df))
    width = 0.35

    plt.bar(
        x - width / 2,
        results_df["Top1_Accuracy"],
        width,
        label="Top-1 Accuracy"
    )

    plt.bar(
        x + width / 2,
        results_df["Student_Test_Macro_F1"],
        width,
        label="Macro F1"
    )

    plt.xticks(
        x,
        results_df["Model"],
        rotation=20
    )

    plt.ylabel("Score")
    plt.title("Student-Test Performance")
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "student_test_comparison.png")
    plt.close()

    # Plot 3: Log Loss
    plt.figure(figsize=(10, 6))

    plt.bar(
        results_df["Model"],
        results_df["Log_Loss"]
    )

    plt.ylabel("Log Loss")
    plt.title("Log Loss Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "log_loss.png")
    plt.close()

    print("\nPlots saved in:")
    print(PLOTS_DIR)


# ==================================================
# 13. MAIN EXECUTION
# ==================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run evaluation with artificial data."
    )

    args = parser.parse_args()

    if args.demo:

        print("Running demonstration evaluation.")

        train, train_labels = create_fake_data(10)
        test, test_labels = create_fake_data(5)

    else:

        print("Loading actual candidate datasets.")

        train, test, train_labels, test_labels = load_data()

    # Validate
    train_features = validate_data(
        train, "Training data"
    )

    test_features = validate_data(
        test, "Student test data"
    )

    if train_features != test_features:
        raise ValueError(
            "Training and test feature columns do not match."
        )

    train_labels = validate_reference_labels(
        train_labels, "Training reference labels"
    )

    test_labels = validate_reference_labels(
        test_labels, "Student test reference labels"
    )

    # Prepare training and test data
    X_train, y_train, groups = prepare_data(
        train, train_features
    )

    X_test, y_test, test_groups = prepare_data(
        test, test_features
    )

    print("\nDATASET SUMMARY")
    print("-" * 50)

    print(f"Training rows: {len(X_train)}")
    print(f"Student-test rows: {len(X_test)}")
    print(f"Features: {train_features}")
    print(f"Training trips: {groups.nunique()}")
    print(f"Student-test trips: {test_groups.nunique()}")

    # Verify cross-validation groups
    check_group_kfold(
        X_train,
        y_train,
        groups
    )

    # Evaluate five models
    results_df = evaluate_models(
        X_train,
        y_train,
        groups,
        X_test,
        y_test,
        test,
        test_labels
    )

    # Generate plots
    generate_plots(results_df)

    # Display comparison
    print("\nFINAL MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df[
            [
                "Model",
                "CV_Macro_F1_Mean",
                "Student_Test_Macro_F1",
                "Top1_Accuracy",
                "Log_Loss"
            ]
        ].to_string(index=False)
    )

    print("\nPhase 3 evaluation completed!")


if __name__ == "__main__":
    main()
