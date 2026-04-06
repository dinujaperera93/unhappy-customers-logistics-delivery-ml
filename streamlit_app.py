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

st.title("Why Are Customers Unhappy?")
st.markdown(
    "This tool identifies the **key drivers of customer unhappiness** in a logistics and delivery survey. "
    "Use the chart below to see which factors matter most, then test individual responses with the predictor."
)

# --- Key drivers (always visible, this is the main insight) ---
st.subheader("Key Drivers of Unhappiness")
importances = model.feature_importances_
labels = list(TAG_TO_QUESTION.values())
keys = list(TAG_TO_QUESTION.keys())
highlight = {"X4", "X5"}
colors = ["#C0392B" if keys[i] in highlight else "#AAB7B8" for i in range(len(keys))]

order = np.argsort(importances)

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([labels[i] for i in order], importances[order], color=[colors[i] for i in order])
ax.set_xlabel("Importance")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
st.pyplot(fig)

st.caption(
    "Satisfaction with courier and price perception are the strongest predictors of unhappiness. "
    "App usability and order completeness have low predictive weight."
)

st.divider()

# --- Individual predictor (secondary) ---
st.subheader("Predict a Single Customer")
st.markdown("Simulate a customer's survey responses to see how the model classifies them.")

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
