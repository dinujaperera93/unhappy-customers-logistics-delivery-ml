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
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
from pathlib import Path


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
ROOT = Path.cwd() # Find the project root directory
DATA_PATH = ROOT / "data" / "ACME-HappinessSurvey2020.csv"

# %%
df_customer = load_data(DATA_PATH)

# %%
_ = explore_data(df_customer)

# %%
SEED = 7964
X_train, X_test, y_train, y_test = split_data(df_customer, "Y", SEED, test_size=0.2)

# %%
models, predictions = select_model(X_train, X_test, y_train, y_test)

# %%
best_model, best_params,_ = tune_hyperparameters(X_train, y_train, SEED)

# %%
clf_report = evaluate_model(best_model, X_test, y_test)

# %%
feature_df = important_features(X_train, best_model, tag_to_comment)

# %%
_,_,removed_features,_ = feature_selection(X_train, y_train, X_test, y_test, best_params, SEED)
