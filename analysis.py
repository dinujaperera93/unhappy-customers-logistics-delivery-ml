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
import os
os.environ["TQDM_DISABLE"] = "1"

from unittest.mock import MagicMock
import sys
sys.modules['tqdm.notebook'] = MagicMock()
sys.modules['tqdm.auto'] = MagicMock()

# %%
from pathlib import Path
from IPython.display import display


from src import (
    load_data,
    explore_data,
    split_data,
    EDA,
    select_model,
    compare_ensembles,
    tune_hyperparameters,
    evaluate_model,
    important_features,
    feature_selection,
    tag_to_comment
)

# %%
ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "ACME-HappinessSurvey2020.csv"

# %%
df = load_data(DATA_PATH)

# %%
explore_data(df)

# %%
SEED = 7964
X_train, X_test, y_train, y_test = split_data(df, "Y", SEED, test_size=0.2)

# %%
EDA(X_train, y_train)

# %%
# LazyPredict (exploration only, uses X_test causing a data leakage)
models, _ = select_model(X_train, X_test, y_train, y_test)
print(models)

# %%
fitted_models, results_df = compare_ensembles(
    X_train, y_train, X_test, y_test, seed=SEED, cv=5
)
results_df

# %%
# Hyperopt tuning for LGBM
best_model, best_params, best_score = tune_hyperparameters(X_train, y_train, SEED)
best_params, round(best_score, 3)

# %%
# Final evaluation of tuned LGBM on test set
print(evaluate_model(best_model, X_test, y_test))

# %%
feature_df = important_features(X_train, best_model, tag_to_comment)
feature_df

# %%
clf_fs, selected_features, removed_features, test_accuracy = feature_selection(
    X_train, y_train, X_test, y_test, best_params, SEED
)

test_accuracy, removed_features

# %%
print("Questions that can be removed from the survey:")
for f in removed_features:
    print("-", tag_to_comment.get(f, f))

# %%
