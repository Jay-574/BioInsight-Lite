import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="BioInsight Lite - Bioactivity Predictor",
    layout="wide"
)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    lr_path = os.path.join(BASE_DIR, "logistic_model_500000.pkl")
    nn_path = os.path.join(BASE_DIR, "nn_model.keras")
    imputer_path = os.path.join(BASE_DIR, "nn_imputer.pkl")
    scaler_path = os.path.join(BASE_DIR, "nn_scaler.pkl")

    if not all(os.path.exists(p) for p in [lr_path, nn_path, imputer_path, scaler_path]):
        st.error("One or more model files are missing in the deployment folder.")
        st.stop()

    lr_model = joblib.load(lr_path)
    nn_model = tf.keras.models.load_model(nn_path)
    nn_imputer = joblib.load(imputer_path)
    nn_scaler = joblib.load(scaler_path)

    return lr_model, nn_model, nn_imputer, nn_scaler


lr_model, nn_model, nn_imputer, nn_scaler = load_models()

# -----------------------------
# Load dataset (optional)
# -----------------------------
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "model_dataset.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path, nrows=50000)


df = load_data()

FEATURES = [
    "molecular_weight",
    "alogp",
    "hba",
    "hbd",
    "tpsa",
    "rotatable_bonds",
    "heavy_atom_count",
    "aromatic_rings"
]

# -----------------------------
# App UI
# -----------------------------
st.title("🧬 BioInsight Lite - Bioactivity Explorer & Predictor")
st.markdown(
    "Predict compound bioactivity using **Logistic Regression** and a "
    "**Neural Network** trained on ChEMBL 36."
)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Data Exploration", "🔮 Prediction", "🧠 Explainability"]
)

# ======================================================
# TAB 1 — DATA EXPLORATION
# ======================================================
with tab1:
    if df.empty:
        st.warning("Dataset not available. Data exploration is disabled.")
    else:
        st.subheader("Dataset Overview")
        st.write(df.head())

        if "active" in df.columns:
            st.subheader("Class Distribution")
            st.bar_chart(df["active"].value_counts())
        else:
            st.warning("'active' column not found in dataset.")

        st.subheader("Feature Statistics")
        available_features = [f for f in FEATURES if f in df.columns]
        st.write(df[available_features].describe())

# ======================================================
# TAB 2 — PREDICTION
# ======================================================
with tab2:
    st.subheader("Single Compound Prediction")

    cols = st.columns(4)
    user_input = {}

    for i, feature in enumerate(FEATURES):
        with cols[i % 4]:
            default_val = float(df[feature].median()) if feature in df.columns else 0.0
            user_input[feature] = st.number_input(
                feature,
                value=default_val
            )

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Bioactivity"):
        # Logistic Regression
        lr_pred = lr_model.predict(input_df)[0]
        lr_prob = lr_model.predict_proba(input_df)[0, 1]

        # Neural Network
        X_nn = nn_imputer.transform(input_df)
        X_nn = nn_scaler.transform(X_nn)
        nn_prob = nn_model.predict(X_nn, verbose=0)[0][0]
        nn_pred = int(nn_prob >= 0.5)

        st.markdown("### Results")

        st.write("**Logistic Regression**")
        st.write(f"Prediction: {'Active' if lr_pred else 'Inactive'}")
        st.write(f"Probability: {lr_prob:.3f}")

        st.write("**Neural Network**")
        st.write(f"Prediction: {'Active' if nn_pred else 'Inactive'}")
        st.write(f"Probability: {nn_prob:.3f}")

# ======================================================
# TAB 3 — EXPLAINABILITY
# ======================================================
with tab3:
    st.subheader("Model Explainability (SHAP)")

    shap_lr = os.path.join(BASE_DIR, "shap_logistic_summary.png")
    shap_nn = os.path.join(BASE_DIR, "shap_nn_summary.png")

    if os.path.exists(shap_lr):
        st.markdown("### Logistic Regression SHAP Summary")
        st.image(shap_lr, use_container_width=True)
    else:
        st.warning("Logistic Regression SHAP image not found.")

    if os.path.exists(shap_nn):
        st.markdown("### Neural Network SHAP Summary")
        st.image(shap_nn, use_container_width=True)
    else:
        st.warning("Neural Network SHAP image not found.")

    st.markdown(
        """
        **Interpretation**:
        - Heavy atom count and lipophilicity strongly influence bioactivity  
        - Excessive polarity and molecular flexibility reduce activity  
        - Neural networks capture smooth non-linear feature interactions  
        """
    )
