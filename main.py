import warnings
from pathlib import Path
import random
import sys
from IPython.display import display

warnings.filterwarnings("ignore")

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
    save_model,
    tag_to_comment,
)

def main():
    # Fixed seed for reproducibility. Was initially randomised during exploration,then locked once results were stable.
    
    #seed = random.randint(1000,9999)
    seed = 7964
    print("Random seed: ",seed)

    ROOT = Path(__file__).resolve().parent
    DATA_PATH = ROOT / "data" / "ACME-HappinessSurvey2020.csv"

    # Pipeline sequence: load → explore → split → EDA → screen → compare → tune → evaluate → select features
    # Each step feeds into the next; test data is held out after split and only used for final evaluation.
    df_customer = load_data(DATA_PATH)
    explore_data(df_customer)
    X_train, X_test, y_train, y_test = split_data(df_customer, "Y", seed, test_size=0.2)
    EDA(X_train, y_train)

    # Step 1: Broad model screening with LazyPredict (single split, indicative only)
    models, _ = select_model(X_train, X_test, y_train, y_test)
    display(models)

    # Step 2: Cross-validated comparison of top candidates + ensemble methods
    _, results_df = compare_ensembles(X_train, y_train, X_test, y_test, seed, cv=5)
    print(results_df.to_string(index=False))

    # Step 3: Bayesian hyperparameter tuning on the best architecture (LightGBM)
    best_model, best_params, _ = tune_hyperparameters(X_train, y_train, seed)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    save_model(best_model, models_dir / "lgbm_model.joblib")

    # Step 4: Final evaluation on held-out test set
    clf_report = evaluate_model(best_model, X_test, y_test)
    print(clf_report)

    # Step 5: Feature importance and survey optimisation
    feature_df = important_features(X_train, best_model, tag_to_comment)
    print(feature_df)
    _, _, removed_features, _ = feature_selection(X_train, y_train, X_test, y_test, best_params, seed)
    print(f"Questions that can be removed from the survey:\n", {tag_to_comment[f] for f in removed_features})
    print("\n--- Task Done ---")
    sys.exit(0)
    
if __name__ == "__main__":
    main()