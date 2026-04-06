import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_PATH = Path("models/lgbm_model.joblib")

TAG_TO_QUESTION = {
    "X1": "Order delivered on time",
    "X2": "Contents of the order was as expected",
    "X3": "Ordered everything wanted to order",
    "X4": "Paid a good price for the order",
    "X5": "Satisfied with courier",
    "X6": "The app makes ordering easy",
}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Customer Happiness Predictor")

inputs = {
    code: st.slider(f"{code} — {question}", min_value=1, max_value=5, value=3)
    for code, question in TAG_TO_QUESTION.items()
}

if st.button("Predict"):
    pred = model.predict(pd.DataFrame([inputs]))[0]
    if pred == 0:
        st.error("Unhappy customer")
    else:
        st.success("Happy customer")

with st.expander("View feature importance"):
    importances = model.feature_importances_
    labels = [f"{code}: {q}" for code, q in TAG_TO_QUESTION.items()]
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([labels[i] for i in order], importances[order], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
