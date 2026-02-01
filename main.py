import sys
import random
import os
import warnings
# Silence warnings
warnings.filterwarnings("ignore")
# Silence LightGBM
os.environ["LIGHTGBM_VERBOSE"] = "-1"
os.environ["LIGHTGBM_LOG_LEVEL"] = "fatal"
# Silence tqdm progress bars (used by LazyPredict)
os.environ["TQDM_DISABLE"] = "1"
from pathlib import Path
from src import load_data, explore_data, split_data, EDA, select_model, compare_ensembles, tune_hyperparameters, evaluate_model, important_features, feature_selection, tag_to_comment


def main():
    #seed = random.randint(1000,9999)
    seed = 7964
    print("Random seed: ",seed)
    
    ROOT = Path(__file__).resolve().parent
    DATA_PATH = ROOT / "data" / "ACME-HappinessSurvey2020.csv"

    df_customer = load_data(DATA_PATH)
    explore_data(df_customer)    
    X_train, X_test, y_train, y_test = split_data(df_customer, "Y", seed, test_size=0.2)
    EDA(X_train,y_train)
    models,_ = select_model(X_train, X_test, y_train, y_test) 
    fitted_models, results_df = compare_ensembles(X_train, y_train, X_test, y_test, seed, cv=5)       
       
    best_model, best_params,_ = tune_hyperparameters(X_train, y_train, seed)
    clf_report = evaluate_model(best_model, X_test, y_test)
    print(clf_report)
    feature_df = important_features(X_train, best_model, tag_to_comment)
    print(feature_df)
    _,_,removed_features,_ = feature_selection(X_train, y_train, X_test, y_test, best_params, seed)
    print(f"Questions that can be removed from the survey:\n",{tag_to_comment[f] for f in removed_features})  
    print("\n--- Task Done ---")
    sys.exit(0)
    
if __name__ == "__main__":
    main()