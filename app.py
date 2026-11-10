
"""Streamlit interface for the Diabetes Risk Analysis project."""

from pathlib import Path

import streamlit as st

from src.predict import load_model_bundle, predict_patient


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "disease_model.joblib"

st.set_page_config(
    page_title="Diabetes Risk Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 2. Load the trained model bundle
# --------------------------------------------------

@st.cache_resource
def get_model_bundle(model_path):
    """Cache the validated model bundle between Streamlit reruns."""
    return load_model_bundle(model_path)


try:
    model_bundle = get_model_bundle(MODEL_PATH)
    threshold = float(model_bundle["threshold"])
    model_name = model_bundle["model_name"]

except Exception as error:
    st.error(
        "The model could not be loaded. "
        "Run the training script and check the model file."
    )
    st.code(str(error))
    st.stop()


# --------------------------------------------------
# 3. Page heading and safety notice
# --------------------------------------------------

st.title("🩺 Diabetes Risk Analysis")
st.caption(
    "An educational machine learning application "
    "using the UCI CDC Diabetes Health Indicators dataset."
)

st.warning(
    "**Educational use only.** This application is not a diagnostic "
    "tool. Its predictions are not clinically validated, and its model "
    "scores are not calibrated individual medical risk probabilities. "
    "Do not use the result to diagnose, treat, or rule out diabetes."
)


# --------------------------------------------------
# 4. Sidebar information
# --------------------------------------------------

with st.sidebar:
    st.header("About this project")

    st.write(
        "This application uses a machine learning model "
        "trained on survey-based health indicators."
    )

    st.write(f"**Selected model:** {model_name}")
    st.write(f"**Decision threshold:** {threshold:.4f}")

    st.divider()
    st.subheader("Target definition")

    st.write(
        "The positive class represents the combined "
        "prediabetes-or-diabetes label in the dataset."
    )

    st.write(
        "The model does not distinguish prediabetes "
        "from diabetes and does not predict future disease onset."
    )

    st.divider()
    st.caption(
        "No patient form data is intentionally saved "
        "by this application."
    )


# --------------------------------------------------
# 5. Dataset category mappings
# --------------------------------------------------

AGE_OPTIONS = {
    "18–24": 1,
    "25–29": 2,
    "30–34": 3,
    "35–39": 4,
    "40–44": 5,
    "45–49": 6,
    "50–54": 7,
    "55–59": 8,
    "60–64": 9,
    "65–69": 10,
    "70–74": 11,
    "75–79": 12,
    "80 or older": 13,
}

EDUCATION_OPTIONS = {
    "Never attended school or kindergarten only": 1,
    "Grades 1–8": 2,
    "Grades 9–11": 3,
    "Grade 12 or GED": 4,
    "College: 1–3 years": 5,
    "College: 4 years or more": 6,
}

INCOME_OPTIONS = {
    "Less than $10,000": 1,
    "$10,000–$14,999": 2,
    "$15,000–$19,999": 3,
    "$20,000–$24,999": 4,
    "$25,000–$34,999": 5,
    "$35,000–$49,999": 6,
    "$50,000–$74,999": 7,
    "$75,000 or more": 8,
}

GENERAL_HEALTH_OPTIONS = {
    "Excellent": 1,
    "Very good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5,
}


# --------------------------------------------------
# 6. Patient input form
# --------------------------------------------------

st.header("Enter health information")

st.write(
    "Enter the available information below. "
    "The categories reflect the original dataset's coding "
    "rather than a clinical assessment."
)

with st.form("patient_information_form"):

    # Demographics
    st.subheader("1. Demographics")
    col1, col2 = st.columns(2)

    with col1:
        sex_label = st.selectbox(
            "Sex recorded in the dataset",
            options=["Female", "Male"],
            help=(
                "This field follows the dataset's binary coding. "
                "It does not capture all gender identities or "
                "biological characteristics."
            ),
        )

        age_label = st.selectbox(
            "Age group",
            options=list(AGE_OPTIONS),
            index=5,
        )

        bmi = st.number_input(
            "BMI",
            min_value=12,
            max_value=98,
            value=28,
            step=1,
            help="Enter BMI as a numeric value.",
        )

    with col2:
        education_label = st.selectbox(
            "Education level",
            options=list(EDUCATION_OPTIONS),
            index=5,
        )

        income_label = st.selectbox(
            "Annual household income category",
            options=list(INCOME_OPTIONS),
            index=5,
            help=(
                "These are the dataset's original US dollar income "
                "categories, not current income bands."
            ),
        )

    # Medical history
    st.subheader("2. Medical history")
    col1, col2 = st.columns(2)

    with col1:
        high_bp = st.selectbox(
            "Ever told you have high blood pressure?",
            ["No", "Yes"],
        )

        high_chol = st.selectbox(
            "Ever told you have high cholesterol?",
            ["No", "Yes"],
        )

        chol_check = st.selectbox(
            "Cholesterol checked within the past 5 years?",
            ["No", "Yes"],
        )

        stroke = st.selectbox(
            "History of stroke?",
            ["No", "Yes"],
        )

    with col2:
        heart_disease = st.selectbox(
            "History of coronary heart disease or heart attack?",
            ["No", "Yes"],
        )

        difficulty_walking = st.selectbox(
            "Serious difficulty walking or climbing stairs?",
            ["No", "Yes"],
        )

        general_health = st.select_slider(
            "General health",
            options=list(GENERAL_HEALTH_OPTIONS),
            value="Good",
        )

    # Lifestyle indicators
    st.subheader("3. Lifestyle and access to care")
    col1, col2 = st.columns(2)

    with col1:
        smoker = st.selectbox(
            "Smoked at least 100 cigarettes in lifetime?",
            ["No", "Yes"],
        )

        physical_activity = st.selectbox(
            "Any physical activity in the past 30 days?",
            ["No", "Yes"],
        )

        fruits = st.selectbox(
            "Consume fruit one or more times per day?",
            ["No", "Yes"],
        )

        vegetables = st.selectbox(
            "Consume vegetables one or more times per day?",
            ["No", "Yes"],
        )

    with col2:
        heavy_alcohol = st.selectbox(
            "Heavy alcohol consumption indicator?",
            ["No", "Yes"],
            help=(
                "Uses the dataset's original indicator; "
                "it is not a general measure of alcohol use."
            ),
        )

        healthcare = st.selectbox(
            "Have any kind of healthcare coverage?",
            ["No", "Yes"],
        )

        cost_barrier = st.selectbox(
            "Unable to see a doctor because of cost in "
            "the past 12 months?",
            ["No", "Yes"],
        )

    # Recent health status
    st.subheader("4. Recent health status")
    col1, col2 = st.columns(2)

    with col1:
        mental_health_days = st.slider(
            "Days mental health was not good in the past 30 days",
            min_value=0,
            max_value=30,
            value=0,
        )

    with col2:
        physical_health_days = st.slider(
            "Days physical health was not good in the past 30 days",
            min_value=0,
            max_value=30,
            value=0,
        )

    submitted = st.form_submit_button(
        "Analyze",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# 7. Convert form answers to dataset encodings
# --------------------------------------------------

def build_patient_inputs():
    """Convert the visible form choices to original dataset codes."""

    return {
        "HighBP": 1 if high_bp == "Yes" else 0,
        "HighChol": 1 if high_chol == "Yes" else 0,
        "CholCheck": 1 if chol_check == "Yes" else 0,
        "BMI": int(bmi),
        "Smoker": 1 if smoker == "Yes" else 0,
        "Stroke": 1 if stroke == "Yes" else 0,
        "HeartDiseaseorAttack": 1 if heart_disease == "Yes" else 0,
        "PhysActivity": 1 if physical_activity == "Yes" else 0,
        "Fruits": 1 if fruits == "Yes" else 0,
        "Veggies": 1 if vegetables == "Yes" else 0,
        "HvyAlcoholConsump": 1 if heavy_alcohol == "Yes" else 0,
        "AnyHealthcare": 1 if healthcare == "Yes" else 0,
        "NoDocbcCost": 1 if cost_barrier == "Yes" else 0,
        "GenHlth": GENERAL_HEALTH_OPTIONS[general_health],
        "MentHlth": int(mental_health_days),
        "PhysHlth": int(physical_health_days),
        "DiffWalk": 1 if difficulty_walking == "Yes" else 0,
        "Sex": 0 if sex_label == "Female" else 1,
        "Age": AGE_OPTIONS[age_label],
        "Education": EDUCATION_OPTIONS[education_label],
        "Income": INCOME_OPTIONS[income_label],
    }


# --------------------------------------------------
# 8. Generate and display a prediction
# --------------------------------------------------

if submitted:
    try:
        patient_inputs = build_patient_inputs()

        patient_df, model_score, predicted_class = predict_patient(
            patient_inputs,
            model_bundle,
        )

        st.divider()
        st.header("Model output")

        score_col, result_col = st.columns(2)

        with score_col:
            st.metric(
                "Positive-class model score",
                f"{model_score * 100:.1f}%",
            )

            st.progress(model_score)

            st.caption(
                "This score is not a calibrated probability "
                "of having or developing diabetes."
            )

        with result_col:
            if predicted_class == 1:
                st.warning("Model classification: Positive class")
            else:
                st.info("Model classification: Negative class")

            st.write(f"Decision threshold: {threshold:.4f}")

        st.subheader("What this result means")

        if predicted_class == 1:
            st.write(
                "The model score meets or exceeds the experimental "
                "classification threshold for the dataset's combined "
                "prediabetes-or-diabetes label."
            )
        else:
            st.write(
                "The model score is below the experimental "
                "classification threshold. This does not rule out "
                "diabetes or prediabetes."
            )

        st.info(
            "This result is based on patterns learned from survey data. "
            "It is not a medical diagnosis, a substitute for laboratory "
            "testing, or a validated estimate of individual disease risk. "
            "Consult a qualified healthcare professional for appropriate "
            "screening and interpretation."
        )

        with st.expander("Review encoded model inputs"):
            st.write(
                "These are the numerical values supplied to the trained model."
            )
            st.dataframe(patient_df, use_container_width=True)

    except Exception as error:
        st.error(
            "Prediction failed. Check the model bundle "
            "and input feature configuration."
        )
        st.code(str(error))


# --------------------------------------------------
# 9. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Academic machine learning project | "
    "UCI CDC Diabetes Health Indicators dataset | "
    "Not intended for clinical decision-making."
)
