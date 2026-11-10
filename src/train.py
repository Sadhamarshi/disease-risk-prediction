
"""Train, evaluate, and save the diabetes classification model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "Diabetes_binary"
TARGET_RECALL = 0.75
RANDOM_STATE = 42

THRESHOLD_CANDIDATES = [
    0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.5168705002126148,
    0.55, 0.60, 0.70, 0.80, 0.90,
]


def evaluate_model(model, X, y, threshold=0.5):
    """Calculate classification and ranking metrics."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(
            y, probabilities
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def choose_threshold(y_true, probabilities, target_recall):
    """Choose the highest-precision threshold meeting target recall."""
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        return 0.5, False

    precision = precision[:-1]
    recall = recall[:-1]
    eligible = np.flatnonzero(recall >= target_recall)

    if len(eligible) == 0:
        best_index = int(np.argmax(recall))
        return float(thresholds[best_index]), False

    best_index = eligible[np.argmax(precision[eligible])]
    return float(thresholds[best_index]), True


def calculate_threshold_metrics(y_true, probabilities, threshold):
    """Calculate threshold-dependent metrics for a fixed score threshold."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(
            y_true, predicted, zero_division=0
        ),
        "recall": recall_score(
            y_true, predicted, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, predicted, zero_division=0
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": (
            float(fp / (fp + tn)) if (fp + tn) else 0.0
        ),
    }


def save_prediction_report(path, y_true, probabilities, threshold):
    """Save actual labels, scores, and classifications to CSV."""
    predicted = (probabilities >= threshold).astype(int)
    pd.DataFrame({
        "actual": np.asarray(y_true, dtype=int),
        "predicted": predicted,
        "estimated_probability": np.asarray(probabilities, dtype=float),
    }).to_csv(path, index=False)


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = (
        project_root
        / "data"
        / "raw"
        / "cdc_diabetes_health_indicators.csv"
    )
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    if df.isnull().any().any():
        raise ValueError(
            "Missing values detected. Review preprocessing first."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected a binary target containing 0 and 1.")

    # Keep identical feature combinations in the same split.
    groups = pd.util.hash_pandas_object(
        X, index=False
    ).to_numpy()

    print("=" * 65)
    print("THREE-WAY GROUPED MODEL TRAINING")
    print("=" * 65)
    print("Dataset shape:", df.shape)
    print("Unique feature groups:", len(np.unique(groups)))

    outer_splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    development_idx, test_idx = next(
        outer_splitter.split(X, y, groups=groups)
    )

    X_development = X.iloc[development_idx]
    y_development = y.iloc[development_idx]
    development_groups = groups[development_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    test_groups = groups[test_idx]

    inner_splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1
    )
    train_local_idx, validation_local_idx = next(
        inner_splitter.split(
            X_development,
            y_development,
            groups=development_groups,
        )
    )

    X_train = X_development.iloc[train_local_idx]
    y_train = y_development.iloc[train_local_idx]
    X_validation = X_development.iloc[validation_local_idx]
    y_validation = y_development.iloc[validation_local_idx]

    train_groups = development_groups[train_local_idx]
    validation_groups = development_groups[validation_local_idx]

    assert not (set(train_groups) & set(validation_groups))
    assert not (set(train_groups) & set(test_groups))
    assert not (set(validation_groups) & set(test_groups))

    print("\nSplit sizes:")
    print("Training:", len(X_train))
    print("Validation:", len(X_validation))
    print("Test:", len(X_test))

    print("\nPositive-class proportions:")
    print("Training:", round(y_train.mean(), 4))
    print("Validation:", round(y_validation.mean(), 4))
    print("Test:", round(y_test.mean(), 4))
    print("Feature-group overlap: none")

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    fitted_models = {}
    validation_results = []

    for name, model in models.items():
        print("\n" + "=" * 65)
        print("Training:", name)
        print("=" * 65)

        model.fit(X_train, y_train)
        fitted_models[name] = model
        result = evaluate_model(model, X_validation, y_validation)

        validation_results.append({
            "model": name,
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1_score": result["f1_score"],
            "roc_auc": result["roc_auc"],
            "average_precision": result["average_precision"],
        })

        print(
            "Validation average precision:",
            round(result["average_precision"], 4),
        )
        print("Validation ROC-AUC:", round(result["roc_auc"], 4))

    validation_df = pd.DataFrame(validation_results).sort_values(
        "average_precision", ascending=False
    )
    validation_df.to_csv(
        reports_dir / "validation_model_comparison.csv",
        index=False,
    )

    print("\nValidation model comparison:")
    print(validation_df.to_string(index=False))

    # Select model using validation average precision only.
    best_model_name = validation_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    print("\nSelected candidate:", best_model_name)

    # Select threshold using validation data only.
    validation_probabilities = best_model.predict_proba(
        X_validation
    )[:, 1]
    threshold, target_met = choose_threshold(
        y_validation, validation_probabilities, TARGET_RECALL
    )
    print("\nThreshold selected on validation data:", threshold)
    print("Target validation recall:", TARGET_RECALL)
    print("Target achieved:", target_met)

    validation_metrics = calculate_threshold_metrics(
        y_validation, validation_probabilities, threshold
    )
    print(
        "Validation recall at selected threshold:",
        round(validation_metrics["recall"], 4),
    )

    # Save validation predictions for threshold analysis.
    save_prediction_report(
        reports_dir / "validation_predictions.csv",
        y_validation,
        validation_probabilities,
        threshold,
    )

    threshold_rows = []
    for candidate in sorted(set(THRESHOLD_CANDIDATES + [threshold])):
        row = calculate_threshold_metrics(
            y_validation, validation_probabilities, candidate
        )
        row["selected_threshold"] = bool(
            np.isclose(candidate, threshold)
        )
        threshold_rows.append(row)

    threshold_df = pd.DataFrame(threshold_rows).sort_values("threshold")
    threshold_df.to_csv(
        reports_dir / "validation_threshold_analysis.csv",
        index=False,
    )

    print("\nValidation threshold analysis:")
    print(threshold_df.round(4).to_string(index=False))

    # Evaluate the selected model once on the held-out test set.
    test_metrics = evaluate_model(
        best_model, X_test, y_test, threshold
    )

    print("\n" + "=" * 65)
    print("FINAL HELD-OUT TEST EVALUATION")
    print("=" * 65)
    print("Selected model:", best_model_name)
    print("Decision threshold:", round(threshold, 6))

    for metric_name in [
        "accuracy", "precision", "recall", "f1_score",
        "roc_auc", "average_precision",
    ]:
        print(f"{metric_name}: {test_metrics[metric_name]:.4f}")

    print("\nTest confusion matrix:")
    print(test_metrics["confusion_matrix"])
    print("\nTest classification report:")
    print(classification_report(
        y_test,
        test_metrics["predictions"],
        labels=[0, 1],
        target_names=["No diabetes", "Prediabetes or diabetes"],
        zero_division=0,
    ))

    test_summary = {
        "model": best_model_name,
        "threshold": threshold,
        "target_validation_recall": TARGET_RECALL,
        "target_recall_met_on_validation": target_met,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1_score": test_metrics["f1_score"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_average_precision": test_metrics["average_precision"],
    }
    pd.DataFrame([test_summary]).to_csv(
        reports_dir / "final_test_metrics.csv", index=False
    )

    save_prediction_report(
        reports_dir / "test_predictions.csv",
        y_test,
        test_metrics["probabilities"],
        threshold,
    )

    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "threshold": threshold,
        "feature_names": X.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "positive_class_description": "Prediabetes or diabetes",
        "target_validation_recall": TARGET_RECALL,
        "target_recall_met_on_validation": target_met,
    }
    model_path = models_dir / "disease_model.joblib"
    joblib.dump(model_bundle, model_path)

    print("\nSaved model bundle:", model_path)
    print("Saved validation comparison:",
          reports_dir / "validation_model_comparison.csv")
    print("Saved validation predictions:",
          reports_dir / "validation_predictions.csv")
    print("Saved validation threshold analysis:",
          reports_dir / "validation_threshold_analysis.csv")
    print("Saved final test metrics:",
          reports_dir / "final_test_metrics.csv")
    print("Saved test predictions:",
          reports_dir / "test_predictions.csv")
    print(
        "\nEducational research model only; not a clinically validated "
        "diagnostic or risk assessment tool."
    )


if __name__ == "__main__":
    main()
