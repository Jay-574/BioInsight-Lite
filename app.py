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

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def get_raw_dataset():
    path = os.path.join(BASE_DIR, "model_dataset.csv")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

st.title("🧬 BioInsight Lite - Bioactivity Explorer & Predictor")
st.markdown(
    "Predict compound bioactivity using **Logistic Regression** and a **Neural Network** trained on ChEMBL 36."
)

csv_data = get_raw_dataset()
if csv_data:
    st.download_button(
        label="Download ChEMBL 36 Dataset (CSV)",
        data=csv_data,
        file_name="model_dataset.csv",
        mime="text/csv"
    )

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

@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "model_dataset.csv")
    if not os.path.exists(csv_path):
         st.warning("model_dataset.csv not found. Exploration tab will be empty.")
         return pd.DataFrame()
    return pd.read_csv(csv_path, nrows=50000)

df = load_data()

FEATURES = [
    "molecular_weight", "alogp", "hba", "hbd",
    "tpsa", "rotatable_bonds",
    "heavy_atom_count", "aromatic_rings"
]

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
    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Class Distribution")
    st.bar_chart(df["active"].value_counts())

    st.subheader("Feature Statistics")
    st.write(df[FEATURES].describe())

# ======================================================
# TAB 2 — PREDICTION
# ======================================================
with tab2:
    st.subheader("Single Compound Prediction")

    cols = st.columns(4)
    user_input = {}

    for i, feature in enumerate(FEATURES):
        with cols[i % 4]:
            user_input[feature] = st.number_input(
                feature,
                value=float(df[feature].median())
            )

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Bioactivity"):
        # Logistic Regression
        lr_pred = lr_model.predict(input_df)[0]
        lr_prob = lr_model.predict_proba(input_df)[0, 1]

        # Neural Network
        X_nn = nn_imputer.transform(input_df)
        X_nn = nn_scaler.transform(X_nn)
        nn_prob = nn_model.predict(X_nn)[0][0]
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

    st.markdown("### Logistic Regression SHAP Summary")
    st.image(os.path.join(BASE_DIR, "shap_logistic_summary.png"), use_container_width=True)

    st.markdown("### Neural Network SHAP Summary")
    st.image(os.path.join(BASE_DIR, "shap_nn_summary.png"), use_container_width=True)

    st.markdown(
        """
        **Interpretation**:
        - Heavy atom count and lipophilicity strongly influence bioactivity
        - Excessive polarity and molecular flexibility reduce activity
        - Neural network captures smoother non-linear interactions
        """
    )
