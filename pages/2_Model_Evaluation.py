
"""Model evaluation, threshold analysis, and error analysis dashboard."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
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
    roc_curve,
)

from src.predict import load_model_bundle


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "disease_model.joblib"
VALIDATION_PATH = ROOT / "reports" / "validation_predictions.csv"
TEST_PATH = ROOT / "reports" / "test_predictions.csv"
METRICS_PATH = ROOT / "reports" / "final_test_metrics.csv"


st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Evaluation & Interpretability")
st.caption(
    "Validation data is used for threshold analysis. "
    "Held-out test data is reserved for final performance and error analysis."
)

st.warning(
    "Educational research only. This model is not a medical diagnostic "
    "tool. Its score is not a calibrated probability of future disease."
)


@st.cache_data
def load_prediction_file(path_string):
    """Load and validate one prediction CSV."""
    path = Path(path_string)

    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)
    required = {
        "actual",
        "predicted",
        "estimated_probability",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {', '.join(sorted(missing))}"
        )
    if df.empty:
        raise ValueError(f"{path.name} is empty.")
    if df[list(required)].isnull().any().any():
        raise ValueError(f"{path.name} contains missing values.")

    for column in ["actual", "predicted"]:
        if not set(df[column].unique()).issubset({0, 1}):
            raise ValueError(f"{column} must contain only 0 and 1.")
        df[column] = df[column].astype(int)

    scores = df["estimated_probability"].to_numpy(dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("Model scores must be finite.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model scores must be between 0 and 1.")

    return df


@st.cache_data
def load_metrics_file(path_string):
    path = Path(path_string)

    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The final test metrics file is empty.")

    return df


@st.cache_resource
def get_model_bundle(path_string):
    return load_model_bundle(path_string)


def calculate_metrics(actual, scores, threshold):
    """Recalculate threshold-dependent metrics."""
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1_score": f1_score(
            actual, predicted, zero_division=0
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": (
            fp / (fp + tn) if (fp + tn) else 0.0
        ),
    }


try:
    validation_df = load_prediction_file(str(VALIDATION_PATH))
    test_df = load_prediction_file(str(TEST_PATH))
    saved_metrics_df = load_metrics_file(str(METRICS_PATH))
    bundle = get_model_bundle(str(MODEL_PATH))
except (FileNotFoundError, ValueError, KeyError, OSError) as error:
    st.error(f"Could not load evaluation files: {error}")
    st.info(
        "Run `python src/train.py` first to generate the validation "
        "predictions and updated reports."
    )
    st.stop()


validation_actual = validation_df["actual"].to_numpy(dtype=int)
validation_scores = validation_df[
    "estimated_probability"
].to_numpy(dtype=float)

test_actual = test_df["actual"].to_numpy(dtype=int)
test_scores = test_df["estimated_probability"].to_numpy(dtype=float)
test_predicted = test_df["predicted"].to_numpy(dtype=int)

threshold = float(bundle["threshold"])

if len(np.unique(validation_actual)) != 2:
    st.error("Validation predictions must contain both target classes.")
    st.stop()

if len(np.unique(test_actual)) != 2:
    st.error("Test predictions must contain both target classes.")
    st.stop()


st.subheader("1. Final held-out test performance")

test_metrics = {
    "Accuracy": accuracy_score(test_actual, test_predicted),
    "Precision": precision_score(
        test_actual, test_predicted, zero_division=0
    ),
    "Recall": recall_score(
        test_actual, test_predicted, zero_division=0
    ),
    "F1-score": f1_score(
        test_actual, test_predicted, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(test_actual, test_scores),
    "Average precision": average_precision_score(
        test_actual, test_scores
    ),
}

cols = st.columns(4)
for col, (label, value) in zip(
    cols,
    [
        ("Test samples", f"{len(test_actual):,}"),
        ("Accuracy", f"{test_metrics['Accuracy']:.2%}"),
        ("Precision", f"{test_metrics['Precision']:.2%}"),
        ("Recall", f"{test_metrics['Recall']:.2%}"),
    ],
):
    col.metric(label, value)

cols = st.columns(4)
for col, (label, value) in zip(
    cols,
    [
        ("F1-score", f"{test_metrics['F1-score']:.4f}"),
        ("ROC-AUC", f"{test_metrics['ROC-AUC']:.4f}"),
        ("Average precision", f"{test_metrics['Average precision']:.4f}"),
        ("Selected threshold", f"{threshold:.4f}"),
    ],
):
    col.metric(label, value)

st.caption(
    f"Selected model: {bundle['model_name']} | "
    f"Target: {bundle.get('target_column', 'Unknown')}"
)

with st.expander("Compare recalculated metrics with the saved report"):
    saved = saved_metrics_df.iloc[0]
    metric_mapping = {
        "Accuracy": "test_accuracy",
        "Precision": "test_precision",
        "Recall": "test_recall",
        "F1-score": "test_f1_score",
        "ROC-AUC": "test_roc_auc",
        "Average precision": "test_average_precision",
    }

    comparison_rows = []
    for label, column in metric_mapping.items():
        if column in saved_metrics_df.columns:
            saved_value = float(saved[column])
            current_value = float(test_metrics[label])
            comparison_rows.append({
                "Metric": label,
                "Saved report": saved_value,
                "Recalculated": current_value,
                "Absolute difference": abs(saved_value - current_value),
            })

    st.dataframe(
        pd.DataFrame(comparison_rows).round(6),
        width="stretch",
        hide_index=True,
    )


st.divider()
st.subheader("2. Interactive threshold analysis")
st.write(
    "Adjust the threshold to see how the classification trade-off changes. "
    "These metrics are calculated from validation predictions, not the test set."
)

selected_threshold = st.slider(
    "Decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(np.clip(threshold, 0.05, 0.95)),
    step=0.01,
)

selected_metrics = calculate_metrics(
    validation_actual,
    validation_scores,
    selected_threshold,
)

cols = st.columns(4)
for col, (label, value) in zip(
    cols,
    [
        ("Precision", f"{selected_metrics['precision']:.2%}"),
        ("Recall", f"{selected_metrics['recall']:.2%}"),
        ("F1-score", f"{selected_metrics['f1_score']:.4f}"),
        ("Accuracy", f"{selected_metrics['accuracy']:.2%}"),
    ],
):
    col.metric(label, value)

cols = st.columns(4)
for col, (label, value) in zip(
    cols,
    [
        ("True positives", f"{selected_metrics['true_positives']:,}"),
        ("False positives", f"{selected_metrics['false_positives']:,}"),
        ("True negatives", f"{selected_metrics['true_negatives']:,}"),
        ("False negatives", f"{selected_metrics['false_negatives']:,}"),
    ],
):
    col.metric(label, value)

st.caption(
    "A lower threshold typically identifies more positive cases, increasing "
    "recall but often increasing false positives. A higher threshold usually "
    "reduces false positives but may miss more positive cases."
)

threshold_values = sorted(set(
    [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, threshold, 0.55, 0.60, 0.70, 0.80, 0.90]
))

threshold_rows = [
    calculate_metrics(
        validation_actual, validation_scores, value
    )
    for value in threshold_values
]

threshold_report = pd.DataFrame(threshold_rows)
threshold_report["selected_threshold"] = np.isclose(
    threshold_report["threshold"], threshold
)

st.markdown("**Threshold comparison table**")
st.dataframe(
    threshold_report.round(4),
    width="stretch",
    hide_index=True,
)

st.download_button(
    "Download validation threshold analysis",
    data=threshold_report.to_csv(index=False).encode("utf-8"),
    file_name="validation_threshold_analysis.csv",
    mime="text/csv",
)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(
    threshold_report["threshold"],
    threshold_report["precision"],
    marker="o",
    label="Precision",
)
ax.plot(
    threshold_report["threshold"],
    threshold_report["recall"],
    marker="o",
    label="Recall",
)
ax.plot(
    threshold_report["threshold"],
    threshold_report["f1_score"],
    marker="o",
    label="F1-score",
)
ax.axvline(
    threshold,
    linestyle="--",
    label=f"Training-selected threshold ({threshold:.3f})",
)
ax.set(
    xlabel="Decision threshold",
    ylabel="Metric score",
    title="Validation metrics by decision threshold",
    ylim=(0, 1.02),
)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


st.divider()
st.subheader("3. Test-set confusion matrix")

test_matrix = confusion_matrix(
    test_actual, test_predicted, labels=[0, 1]
)
tn, fp, fn, tp = test_matrix.ravel()

left, right = st.columns([1, 1])

with left:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    image = ax.imshow(test_matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        xlabel="Predicted class",
        ylabel="Actual class",
        title="Held-out test confusion matrix",
    )
    for row in range(2):
        for col in range(2):
            ax.text(
                col, row, f"{test_matrix[row, col]:,}",
                ha="center", va="center",
            )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.metric("True negatives", f"{tn:,}")
    st.metric("False positives", f"{fp:,}")
    st.metric("False negatives", f"{fn:,}")
    st.metric("True positives", f"{tp:,}")

st.caption(
    "Positive means the dataset's combined prediabetes-or-diabetes label. "
    "It does not mean a clinical diagnosis."
)


st.divider()
st.subheader("4. Error analysis on the held-out test set")

errors_df = test_df.copy()
errors_df["error_type"] = np.select(
    [
        (errors_df["actual"] == 0) & (errors_df["predicted"] == 1),
        (errors_df["actual"] == 1) & (errors_df["predicted"] == 0),
    ],
    ["False positive", "False negative"],
    default="Correct prediction",
)
errors_df["score_distance_from_threshold"] = (
    errors_df["estimated_probability"] - threshold
).abs()

false_positives = errors_df[
    errors_df["error_type"] == "False positive"
]
false_negatives = errors_df[
    errors_df["error_type"] == "False negative"
]

cols = st.columns(3)
cols[0].metric("All test errors", f"{len(false_positives) + len(false_negatives):,}")
cols[1].metric("False positives", f"{len(false_positives):,}")
cols[2].metric("False negatives", f"{len(false_negatives):,}")

error_filter = st.selectbox(
    "Select records to inspect",
    [
        "All misclassified records",
        "False positives only",
        "False negatives only",
        "Correct predictions",
    ],
)

if error_filter == "All misclassified records":
    visible_errors = errors_df[
        errors_df["error_type"] != "Correct prediction"
    ]
elif error_filter == "False positives only":
    visible_errors = false_positives
elif error_filter == "False negatives only":
    visible_errors = false_negatives
else:
    visible_errors = errors_df[
        errors_df["error_type"] == "Correct prediction"
    ]

visible_errors = visible_errors.sort_values(
    "score_distance_from_threshold"
)

st.write(
    f"Showing {len(visible_errors):,} records. Records closest to the "
    "decision threshold appear first."
)

st.dataframe(
    visible_errors.head(500).round(5),
    width="stretch",
    hide_index=True,
)

st.caption(
    "The CSV contains labels and model scores only, not individual feature "
    "values or personally identifying information."
)

st.download_button(
    "Download selected error-analysis records",
    data=visible_errors.to_csv(index=False).encode("utf-8"),
    file_name="test_error_analysis.csv",
    mime="text/csv",
)


st.divider()
st.subheader("5. ROC and precision–recall curves")

left, right = st.columns(2)

fpr, tpr, _ = roc_curve(test_actual, test_scores)
roc_auc = roc_auc_score(test_actual, test_scores)

with left:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", label="Random baseline")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate (recall)",
        title="Test ROC curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

precision_curve, recall_curve, _ = precision_recall_curve(
    test_actual, test_scores
)
average_precision = average_precision_score(test_actual, test_scores)
prevalence = float(np.mean(test_actual))

with right:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        recall_curve,
        precision_curve,
        label=f"Average precision = {average_precision:.4f}",
    )
    ax.axhline(
        prevalence,
        linestyle="--",
        label=f"Positive prevalence = {prevalence:.2%}",
    )
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Test precision–recall curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


st.divider()
st.subheader("6. Classification report")

report = classification_report(
    test_actual,
    test_predicted,
    labels=[0, 1],
    target_names=[
        "Negative: no target label",
        "Positive: prediabetes or diabetes",
    ],
    output_dict=True,
    zero_division=0,
)
st.dataframe(
    pd.DataFrame(report).transpose().round(4),
    width="stretch",
)

st.divider()
st.subheader("Interpretation and limitations")
st.markdown(
    """
    - Threshold analysis uses validation predictions.
    - Final performance and error analysis use held-out test predictions.
    - Changing the slider does not retrain the model or change the saved
      production threshold.
    - False positives are negative-label cases classified as positive.
    - False negatives are positive-label cases classified as negative.
    - The score is not a calibrated probability of future disease.
    - The dataset is observational; model associations do not establish
      causation.
    - This application must not be used for medical diagnosis or treatment.
    """
)

st.success("Model evaluation dashboard loaded successfully.")
