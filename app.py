"""Streamlit demo: Credit Card Customer Segmentation (K-Means, k=2).

Put this file in the repo root, next to the `models/` folder
(models/scaler.pkl, models/kmeans_model.pkl, models/pca.pkl).
Run locally:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).parent / "models"

# 18 raw columns minus CUST_ID, ONEOFF_PURCHASES, PURCHASES_INSTALLMENTS_FREQUENCY
# (the same 15 features used in the notebook, in dataset order).
FEATURES = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE",
]

# The 12 features with skew > 1.0 that got log1p in the notebook.
LOG_COLS = [
    "BALANCE", "PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE",
    "ONEOFF_PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX",
    "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
]

# name: (label, kind, min, max, step)   kind: "money" | "ratio" | "count"
FIELDS = {
    "BALANCE": ("Balance ($)", "money", 0.0, 100000.0, 50.0),
    "BALANCE_FREQUENCY": ("Balance update frequency (0-1)", "ratio", 0.0, 1.0, 0.05),
    "PURCHASES": ("Total purchases ($)", "money", 0.0, 100000.0, 50.0),
    "INSTALLMENTS_PURCHASES": ("Installment purchases ($)", "money", 0.0, 100000.0, 50.0),
    "CASH_ADVANCE": ("Cash advance ($)", "money", 0.0, 100000.0, 50.0),
    "PURCHASES_FREQUENCY": ("Purchase frequency (0-1)", "ratio", 0.0, 1.0, 0.05),
    "ONEOFF_PURCHASES_FREQUENCY": ("One-off purchase frequency (0-1)", "ratio", 0.0, 1.0, 0.05),
    "CASH_ADVANCE_FREQUENCY": ("Cash advance frequency (0-1)", "ratio", 0.0, 1.0, 0.05),
    "CASH_ADVANCE_TRX": ("Cash advance transactions", "count", 0, 500, 1),
    "PURCHASES_TRX": ("Purchase transactions", "count", 0, 500, 1),
    "CREDIT_LIMIT": ("Credit limit ($)", "money", 0.0, 100000.0, 100.0),
    "PAYMENTS": ("Payments made ($)", "money", 0.0, 100000.0, 50.0),
    "MINIMUM_PAYMENTS": ("Minimum payments ($)", "money", 0.0, 100000.0, 50.0),
    "PRC_FULL_PAYMENT": ("Share of months paid in full (0-1)", "ratio", 0.0, 1.0, 0.05),
    "TENURE": ("Tenure (months)", "count", 6, 12, 1),
}

# Demo presets. Balance / purchases / cash advance / payments / full-pay / min-pay
# come from the README's cluster means; the rest are plausible example values.
PRESETS = {
    "Purchase-active example": dict(
        BALANCE=940.0, BALANCE_FREQUENCY=0.95, PURCHASES=1450.0,
        INSTALLMENTS_PURCHASES=650.0, CASH_ADVANCE=120.0, PURCHASES_FREQUENCY=0.7,
        ONEOFF_PURCHASES_FREQUENCY=0.3, CASH_ADVANCE_FREQUENCY=0.05,
        CASH_ADVANCE_TRX=0, PURCHASES_TRX=21, CREDIT_LIMIT=4500.0, PAYMENTS=1575.0,
        MINIMUM_PAYMENTS=645.0, PRC_FULL_PAYMENT=0.23, TENURE=12,
    ),
    "Cash-advance carrier example": dict(
        BALANCE=2520.0, BALANCE_FREQUENCY=0.95, PURCHASES=320.0,
        INSTALLMENTS_PURCHASES=150.0, CASH_ADVANCE=2280.0, PURCHASES_FREQUENCY=0.2,
        ONEOFF_PURCHASES_FREQUENCY=0.1, CASH_ADVANCE_FREQUENCY=0.4,
        CASH_ADVANCE_TRX=8, PURCHASES_TRX=4, CREDIT_LIMIT=4500.0, PAYMENTS=1970.0,
        MINIMUM_PAYMENTS=1150.0, PRC_FULL_PAYMENT=0.04, TENURE=12,
    ),
}

# Cluster 0 / 1 meaning, taken from the README results section.
SEGMENTS = {
    0: dict(
        name="Purchase-active", share="60.4%", kind="success",
        text="Buys regularly, carries moderate balances, rarely uses cash advances.",
        action="Grow with them: rewards, limit increases and point-linked offers.",
    ),
    1: dict(
        name="Cash-advance & balance carrier", share="39.6%", kind="warning",
        text="Carries high balances, leans on cash advances, buys rarely and almost never pays in full.",
        action="Manage risk first: tighter limit policy; balance-transfer products fit this group.",
    ),
}

PROFILE = pd.DataFrame(
    {
        "Purchase-active (0)": ["$938", "$1,452", "$124", "$1,576", "23%", "$646"],
        "Cash-advance carriers (1)": ["$2,519", "$319", "$2,282", "$1,973", "4%", "$1,149"],
    },
    index=["Balance", "Purchases", "Cash advance", "Payments", "Paid in full", "Minimum payments"],
)


@st.cache_resource
def load_models():
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    kmeans = joblib.load(MODELS_DIR / "kmeans_model.pkl")
    pca_path = MODELS_DIR / "pca.pkl"
    pca = joblib.load(pca_path) if pca_path.exists() else None
    return scaler, pca, kmeans


def predict_segment(values, scaler, pca, kmeans):
    """Raw inputs -> log1p on skewed cols -> scale -> (PCA if the model needs it) -> K-Means."""
    names = getattr(scaler, "feature_names_in_", None)
    cols = list(names) if names is not None else FEATURES
    row = pd.DataFrame([values])[cols].astype(float)

    log_cols = [c for c in LOG_COLS if c in cols]
    row[log_cols] = np.log1p(row[log_cols])

    X = scaler.transform(row if names is not None else row.to_numpy())

    # K-Means may have been fit on the scaled features or on the PCA output.
    if kmeans.n_features_in_ != X.shape[1]:
        if pca is None or getattr(pca, "n_components_", None) != kmeans.n_features_in_:
            raise ValueError("K-Means input size does not match the saved scaler/PCA.")
        X = pca.transform(X)
    return int(kmeans.predict(X)[0])


def apply_preset():
    choice = st.session_state["preset"]
    for name, value in PRESETS[choice].items():
        st.session_state[f"in_{name}"] = value


st.set_page_config(page_title="Credit Card Customer Segmentation", page_icon="💳")
st.title("💳 Credit Card Customer Segmentation")
st.caption(
    "K-Means (k=2) trained on 8,950 cardholders and 6 months of behavior. "
    "Enter a customer's usage and see which segment they fall into."
)

try:
    scaler, pca, kmeans = load_models()
except Exception as exc:  # missing files or scikit-learn version mismatch
    st.error(f"Could not load the saved models from `models/`: {exc}")
    st.stop()

if "preset" not in st.session_state:
    st.session_state["preset"] = "Purchase-active example"
    for name, value in PRESETS["Purchase-active example"].items():
        st.session_state[f"in_{name}"] = value

st.selectbox("Load an example customer", list(PRESETS), key="preset", on_change=apply_preset)

values = {}
left, right = st.columns(2)
for i, name in enumerate(FEATURES):
    label, kind, lo, hi, step = FIELDS[name]
    with (left if i % 2 == 0 else right):
        if kind == "ratio":
            values[name] = st.slider(label, lo, hi, step=step, key=f"in_{name}")
        else:
            values[name] = st.number_input(label, min_value=lo, max_value=hi, step=step, key=f"in_{name}")

if st.button("Predict segment", type="primary"):
    try:
        cluster = predict_segment(values, scaler, pca, kmeans)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        seg = SEGMENTS.get(cluster)
        if seg is None:
            st.info(f"Cluster {cluster}")
        else:
            message = f"**Cluster {cluster} — {seg['name']}** ({seg['share']} of customers)"
            (st.success if seg["kind"] == "success" else st.warning)(message)
            st.write(seg["text"])
            st.write(f"**Suggested action:** {seg['action']}")

with st.expander("How the two segments differ (average raw values)"):
    st.table(PROFILE)
    st.caption(
        "Pipeline: log1p on the 12 skewed features → StandardScaler → K-Means. "
        "Source: Kaggle Credit Card Dataset for Clustering."
    )
