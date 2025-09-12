import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.data_preprocess import load_and_preprocess_data
from src.explain import shap_summary, find_best_threshold
from src.evaluate import evaluate_model

# ========================
# CONFIG
# ========================
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# SIDEBAR
# ========================
st.sidebar.title("⚙️ Settings")

DATA_PATH = "data/creditcard.csv"

# Model selection
model_choice = st.sidebar.radio(
    "Choose Model",
    ["XGBoost", "Random Forest"],
    index=0
)

# Options
show_shap = st.sidebar.checkbox("Show SHAP Feature Importance", value=True)
show_threshold = st.sidebar.checkbox("Find Best Threshold (90% Precision)")

# ========================
# MAIN TITLE
# ========================
st.title("💳 Credit Card Fraud Detection Dashboard")

# ========================
# TAB LAYOUT
# ========================
tabs = st.tabs(["📂 Data Preview", "📊 Evaluation", "🔍 Explainability"])

# ========================
# TAB 1 - Dataset
# ========================
with tabs[0]:
    st.header("📂 Dataset Preview")
    try:
        df = pd.read_csv(DATA_PATH)
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        fraud_cases = df[df["Class"] == 1].shape[0]
        valid_cases = df[df["Class"] == 0].shape[0]
        col1, col2 = st.columns(2)
        col1.metric("Fraud Cases", f"{fraud_cases:,}")
        col2.metric("Valid Cases", f"{valid_cases:,}")
        # ========================
        # Preprocessing
        # ========================
        X_train, X_val, X_test, y_train, y_val, y_test, summary = load_and_preprocess_data(DATA_PATH)

        with st.expander("🔧 View Preprocessing Summary"):
            st.json(summary)

    except FileNotFoundError:
        st.error("❌ Dataset not found. Please ensure `data/creditcard.csv` exists.")
        st.stop()



# ========================
# Load Model & Scaler
# ========================
try:
    if model_choice == "XGBoost":
        model = joblib.load("models/fraud_xgb_model.pkl")
    else:
        model = joblib.load("models/fraud_rfc_model.pkl")
    st.sidebar.success(f"✅ Loaded Model: {model_choice}")
except FileNotFoundError:
    st.error(f"❌ Model file for {model_choice} not found. Train & save model first.")
    st.stop()

# If you have a scaler, load it
try:
    scaler = joblib.load("models/scaler.pkl")  # save scaler during training
except:
    scaler = None  # optional if your model is tree-based

# ========================
# TAB 2 - Evaluation
# ========================
with tabs[1]:
    st.header("📊 Model Evaluation")
    evaluate_model(model, X_test, y_test)

# ========================
# TAB 3 - Explainability
# ========================
with tabs[2]:
    st.header("🔍 Explainability")

    if show_shap:
        st.subheader("Feature Importance with SHAP")
        shap_summary(model, X_test)

    if show_threshold:
        st.subheader("Optimal Threshold for Precision ≥ 90%")
        best_threshold = find_best_threshold(model, X_test, y_test)
        st.success(f"🎯 Best Threshold: {best_threshold:.4f}")

