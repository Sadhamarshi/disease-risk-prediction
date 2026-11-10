
"""Reusable prediction functions for the diabetes analysis project."""

from pathlib import Path
from typing import Any, Mapping
import math

import joblib
import pandas as pd


EXPECTED_FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

BINARY_FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]

CATEGORICAL_RANGES = {
    "GenHlth": (1, 5),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}

DAY_FEATURES = ["MentHlth", "PhysHlth"]


def load_model_bundle(model_path: str | Path) -> dict[str, Any]:
    """Load and validate the trained model bundle."""
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise ValueError("The model bundle must be a dictionary.")

    required_keys = {
        "model",
        "model_name",
        "threshold",
        "feature_names",
        "target_column",
    }

    missing_keys = required_keys.difference(bundle)

    if missing_keys:
        raise ValueError(
            "The saved model bundle is missing required keys: "
            + ", ".join(sorted(missing_keys))
        )

    try:
        threshold = float(bundle["threshold"])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("The model threshold must be numeric.") from error

    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("The model threshold must be finite and between 0 and 1.")

    feature_names = list(bundle["feature_names"])

    if feature_names != EXPECTED_FEATURES:
        raise ValueError(
            "The saved model feature names do not match the "
            "expected 21 training features and their order."
        )

    if not callable(getattr(bundle["model"], "predict_proba", None)):
        raise ValueError("The saved model must support predict_proba().")

    bundle["threshold"] = threshold
    bundle["feature_names"] = feature_names

    return bundle


def yes_no(value: str) -> int:
    """Convert a Yes/No form answer to its dataset encoding."""
    if value == "Yes":
        return 1

    if value == "No":
        return 0

    raise ValueError(f"Expected 'Yes' or 'No', received {value!r}.")


def encode_patient_inputs(
    inputs: Mapping[str, Any],
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Validate encoded input values and build one model input row."""
    names = list(
        EXPECTED_FEATURES if feature_names is None else feature_names
    )

    if names != EXPECTED_FEATURES:
        raise ValueError(
            "Feature names must match the expected training order."
        )

    missing = set(names).difference(inputs)
    extra = set(inputs).difference(names)

    if missing:
        raise ValueError(
            "Missing required features: " + ", ".join(sorted(missing))
        )

    if extra:
        raise ValueError(
            "Unexpected features: " + ", ".join(sorted(extra))
        )

    record = {}

    for feature in names:
        value = inputs[feature]

        if value is None or isinstance(value, bool):
            raise ValueError(f"{feature} must be a valid numeric value.")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{feature} must be numeric; received {value!r}."
            ) from error

        # Reject NaN and positive/negative infinity before integer conversion.
        if not math.isfinite(numeric_value):
            raise ValueError(f"{feature} must be finite and numeric.")

        if numeric_value != int(numeric_value):
            raise ValueError(f"{feature} must be a whole number.")

        numeric_value = int(numeric_value)

        if feature in BINARY_FEATURES and numeric_value not in (0, 1):
            raise ValueError(f"{feature} must be 0 or 1.")

        if feature == "BMI" and not 12 <= numeric_value <= 98:
            raise ValueError("BMI must be between 12 and 98.")

        if feature in DAY_FEATURES and not 0 <= numeric_value <= 30:
            raise ValueError(f"{feature} must be between 0 and 30.")

        if feature in CATEGORICAL_RANGES:
            minimum, maximum = CATEGORICAL_RANGES[feature]

            if not minimum <= numeric_value <= maximum:
                raise ValueError(
                    f"{feature} must be between {minimum} and {maximum}."
                )

        record[feature] = numeric_value

    return pd.DataFrame([record], columns=names)


def predict_patient(
    inputs: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> tuple[pd.DataFrame, float, int]:
    """Validate inputs, predict the positive-class score, and classify."""
    feature_names = list(bundle["feature_names"])
    threshold = float(bundle["threshold"])
    model = bundle["model"]

    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("The model threshold must be finite and between 0 and 1.")

    patient_df = encode_patient_inputs(inputs, feature_names)
    probabilities = model.predict_proba(patient_df)

    if (
        getattr(probabilities, "ndim", 0) != 2
        or probabilities.shape != (1, 2)
    ):
        raise ValueError(
            "The model must return two class scores for one input row."
        )

    score = float(probabilities[0, 1])

    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("The model returned an invalid positive-class score.")

    predicted_class = int(score >= threshold)

    return patient_df, score, predicted_class
