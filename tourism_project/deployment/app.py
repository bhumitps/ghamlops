import streamlit as st
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
MODEL_REPO_ID = "bhumitps/tourism_model"
MODEL_FILENAME = "best_tourism_model_v3.joblib"


@st.cache_resource
def load_model():
    """Download the model from HF Hub and load it with joblib."""
    st.write("Loading model from Hugging Face Hub...")
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model",
            force_download=True,  # always fetch latest v3 model
        )
        model = joblib.load(model_path)

        # ------------------------------------------------------------------
        # Patch for xgboost version mismatch:
        # some older models don't have .use_label_encoder,
        # but newer xgboost expects it.
        # ------------------------------------------------------------------
        try:
            from xgboost import XGBClassifier

            # If it's a sklearn Pipeline, grab the final XGBClassifier
            xgb_clf = None
            if hasattr(model, "named_steps"):
                # our pipeline step is named "model"
                xgb_clf = model.named_steps.get("model", None)
            elif isinstance(model, XGBClassifier):
                xgb_clf = model

            if isinstance(xgb_clf, XGBClassifier) and not hasattr(
                xgb_clf, "use_label_encoder"
            ):
                # Safe default for modern xgboost
                xgb_clf.use_label_encoder = False
        except Exception:
            # If anything goes wrong with the patch, don't break the app
            pass

        st.write("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise


model = load_model()

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.title("Wellness Tourism Package Purchase Prediction")

st.write(
    """
    Predict whether a customer is likely to purchase the
    **Wellness Tourism Package**.

    Fill in the customer details below and click **Predict**.
    """
)

# --- Input fields ---
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=0, max_value=100, value=35)
    CityTier = st.selectbox("CityTier", options=[1, 2, 3], index=0)
    DurationOfPitch = st.number_input(
        "DurationOfPitch (minutes)", min_value=0, max_value=300, value=15
    )
    NumberOfPersonVisiting = st.number_input(
        "NumberOfPersonVisiting", min_value=1, max_value=20, value=2
    )
    NumberOfFollowups = st.number_input(
        "NumberOfFollowups", min_value=0, max_value=20, value=2
    )
    PreferredPropertyStar = st.selectbox(
        "PreferredPropertyStar", options=[1, 2, 3, 4, 5], index=2
    )
    NumberOfTrips = st.number_input(
        "NumberOfTrips", min_value=0, max_value=50, value=1
    )
    NumberOfChildrenVisiting = st.number_input(
        "NumberOfChildrenVisiting", min_value=0, max_value=10, value=0
    )
    MonthlyIncome = st.number_input(
        "MonthlyIncome", min_value=0, max_value=1_000_000, value=50_000, step=1000
    )

with col2:
    TypeofContact = st.selectbox(
        "TypeofContact",
        options=["Self Enquiry", "Company Invited", "Other"],
    )
    Occupation = st.selectbox(
        "Occupation", options=["Salaried", "Self Employed", "Free Lancer", "Other"]
    )
    Gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    ProductPitched = st.text_input("ProductPitched (raw value)", value="Basic")
    MaritalStatus = st.selectbox(
        "MaritalStatus", options=["Married", "Single", "Divorced", "Other"]
    )
    Passport = st.selectbox("Passport", options=["No", "Yes"])
    PitchSatisfactionScore = st.selectbox(
        "PitchSatisfactionScore", options=[1, 2, 3, 4, 5], index=2
    )
    OwnCar = st.selectbox("OwnCar", options=["No", "Yes"])
    Designation = st.selectbox(
        "Designation",
        options=["Executive", "Manager", "Senior Manager", "AVP", "VP", "Other"],
    )

st.markdown("---")

# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------
if st.button("Predict"):
    # Map Yes/No to the numeric format used during training (0/1)
    passport_num = 1 if Passport == "Yes" else 0
    owncar_num = 1 if OwnCar == "Yes" else 0

    input_data = pd.DataFrame(
        [
            {
                "Age": Age,
                "TypeofContact": TypeofContact,
                "CityTier": CityTier,
                "DurationOfPitch": DurationOfPitch,
                "Occupation": Occupation,
                "Gender": Gender,
                "NumberOfPersonVisiting": NumberOfPersonVisiting,
                "NumberOfFollowups": NumberOfFollowups,
                "ProductPitched": ProductPitched,
                "PreferredPropertyStar": PreferredPropertyStar,
                "MaritalStatus": MaritalStatus,
                "NumberOfTrips": NumberOfTrips,
                "Passport": passport_num,  # numeric
                "PitchSatisfactionScore": PitchSatisfactionScore,
                "OwnCar": owncar_num,       # numeric
                "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
                "Designation": Designation,
                "MonthlyIncome": MonthlyIncome,
            }
        ]
    )

    # Make prediction
    pred_proba = model.predict_proba(input_data)[0][1]
    pred_label = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    if pred_label == 1:
        st.success(
            f"Customer is **LIKELY** to purchase the Wellness Tourism Package. "
            f"(Probability: {pred_proba:.2%})"
        )
    else:
        st.info(
            f"Customer is **UNLIKELY** to purchase the Wellness Tourism Package. "
            f"(Probability: {pred_proba:.2%})"
        )

    st.caption("Note: probabilities are model-based estimates and not guarantees.")
