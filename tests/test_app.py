
"""Automated tests for the reusable diabetes prediction module."""

from pathlib import Path

import numpy as np
import pytest

from src.predict import (
    EXPECTED_FEATURES,
    encode_patient_inputs,
    load_model_bundle,
    predict_patient,
    yes_no,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "disease_model.joblib"


@pytest.fixture(scope="module")
def bundle():
    """Load the real trained model for integration tests."""
    return load_model_bundle(MODEL_PATH)


@pytest.fixture
def valid_inputs():
    """Return a synthetic, valid input record."""
    return {
        "HighBP": 0,
        "HighChol": 0,
        "CholCheck": 1,
        "BMI": 22,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 1,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": 2,
        "Education": 6,
        "Income": 8,
    }


def test_model_bundle_is_valid(bundle):
    assert bundle["model_name"]
    assert 0.0 <= bundle["threshold"] <= 1.0
    assert bundle["feature_names"] == EXPECTED_FEATURES
    assert bundle["target_column"]


def test_feature_order_matches_training(bundle):
    assert bundle["feature_names"] == EXPECTED_FEATURES
    assert len(set(bundle["feature_names"])) == 21


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("No", 0),
        ("Yes", 1),
    ],
)
def test_yes_no_encoding(answer, expected):
    assert yes_no(answer) == expected


def test_yes_no_rejects_invalid_answer():
    with pytest.raises(ValueError):
        yes_no("Maybe")


def test_encoded_input_has_correct_shape_and_order(valid_inputs):
    patient_df = encode_patient_inputs(valid_inputs)

    assert patient_df.shape == (1, 21)
    assert list(patient_df.columns) == EXPECTED_FEATURES
    assert patient_df.iloc[0].to_dict() == valid_inputs


def test_encoding_uses_supplied_feature_order(valid_inputs):
    patient_df = encode_patient_inputs(
        valid_inputs,
        feature_names=EXPECTED_FEATURES,
    )

    assert list(patient_df.columns) == EXPECTED_FEATURES


def test_missing_feature_is_rejected(valid_inputs):
    incomplete_inputs = valid_inputs.copy()
    del incomplete_inputs["BMI"]

    with pytest.raises(ValueError, match="Missing required features"):
        encode_patient_inputs(incomplete_inputs)


def test_unexpected_feature_is_rejected(valid_inputs):
    invalid_inputs = valid_inputs.copy()
    invalid_inputs["UnknownFeature"] = 1

    with pytest.raises(ValueError, match="Unexpected features"):
        encode_patient_inputs(invalid_inputs)


@pytest.mark.parametrize(
    ("feature", "invalid_value"),
    [
        ("HighBP", 2),
        ("Sex", -1),
        ("BMI", 11),
        ("BMI", 99),
        ("GenHlth", 0),
        ("GenHlth", 6),
        ("MentHlth", -1),
        ("MentHlth", 31),
        ("PhysHlth", -1),
        ("PhysHlth", 31),
        ("Age", 0),
        ("Age", 14),
        ("Education", 7),
        ("Income", 0),
    ],
)
def test_invalid_feature_ranges_are_rejected(
    valid_inputs, feature, invalid_value
):
    invalid_inputs = valid_inputs.copy()
    invalid_inputs[feature] = invalid_value

    with pytest.raises(ValueError):
        encode_patient_inputs(invalid_inputs)


@pytest.mark.parametrize(
    "invalid_value",
    [None, float("nan"), float("inf"), "not a number", True],
)
def test_invalid_bmi_values_are_rejected(valid_inputs, invalid_value):
    invalid_inputs = valid_inputs.copy()
    invalid_inputs["BMI"] = invalid_value

    with pytest.raises(ValueError):
        encode_patient_inputs(invalid_inputs)


def test_prediction_returns_valid_score_and_class(bundle, valid_inputs):
    patient_df, score, predicted_class = predict_patient(
        valid_inputs,
        bundle,
    )

    assert patient_df.shape == (1, 21)
    assert list(patient_df.columns) == EXPECTED_FEATURES
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0
    assert predicted_class in (0, 1)
    assert predicted_class == int(score >= bundle["threshold"])


def test_threshold_logic_below_at_and_above_threshold(bundle, valid_inputs):
    """Test the classification rule at the real configured threshold."""

    threshold = bundle["threshold"]

    class FakeModel:
        def __init__(self, score):
            self.score = score

        def predict_proba(self, dataframe):
            return np.array([[1.0 - self.score, self.score]])

    test_cases = [
        (max(0.0, threshold - 0.001), 0),
        (threshold, 1),
        (min(1.0, threshold + 0.001), 1),
    ]

    for score, expected_class in test_cases:
        fake_bundle = {
            "model": FakeModel(score),
            "threshold": threshold,
            "feature_names": EXPECTED_FEATURES,
        }

        _, returned_score, predicted_class = predict_patient(
            valid_inputs,
            fake_bundle,
        )

        assert returned_score == pytest.approx(score)
        assert predicted_class == expected_class


def test_prediction_rejects_invalid_model_scores(valid_inputs, bundle):
    class FakeModel:
        def predict_proba(self, dataframe):
            return np.array([[0.5, np.nan]])

    fake_bundle = {
        "model": FakeModel(),
        "threshold": bundle["threshold"],
        "feature_names": EXPECTED_FEATURES,
    }

    with pytest.raises(ValueError, match="invalid positive-class score"):
        predict_patient(valid_inputs, fake_bundle)


def test_missing_model_file_raises_error(tmp_path):
    missing_path = tmp_path / "missing_model.joblib"

    with pytest.raises(FileNotFoundError):
        load_model_bundle(missing_path)


def test_invalid_feature_order_is_rejected(valid_inputs):
    wrong_order = EXPECTED_FEATURES.copy()
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]

    with pytest.raises(ValueError, match="training order"):
        encode_patient_inputs(valid_inputs, feature_names=wrong_order)
