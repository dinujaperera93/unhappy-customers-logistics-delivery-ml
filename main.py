import sys
import random
from pathlib import Path
from src import load_data, explore_data,split_data, select_model, tune_hyperparameters, evaluate_model, important_features,feature_selection, tag_to_comment


def main():
    #seed = random.randint(1000,9999)
    seed = 7964
    print("Random seed: ",seed)
    
    ROOT = Path(__file__).resolve().parent
    DATA_PATH = ROOT / "data" / "ACME-HappinessSurvey2020.csv"

    df_customer = load_data(DATA_PATH)
    explore_df  = explore_data(df_customer)
    X_train, X_test, y_train, y_test = split_data(explore_df, "Y", seed, test_size=0.2)
    
    models,_ = select_model(X_train, X_test, y_train, y_test)    
       
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