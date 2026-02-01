# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %%
from pathlib import Path
import pandas as pd

from src import (
    load_data,
    explore_data,
    split_data,
    select_model,
    tune_hyperparameters,
    evaluate_model,
    important_features,
    feature_selection,
    tag_to_comment,
)

# %%
ROOT = Path(__file__).resolve().parent # Find the project root directory
DATA_PATH = ROOT / "data" / "ACME-HappinessSurvey2020.csv"

# %%
df_customer = load_data(DATA_PATH)
df_customer.head()

# %%
_ = explore_data(df_customer)

# %%
SEED = 7964
X_train, X_test, y_train, y_test = split_data(df_customer, "Y", SEED, test_size=0.2)
