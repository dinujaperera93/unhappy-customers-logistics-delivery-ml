import sys
from customer_model import load_data, explore_data,split_data, tune_hyperparameters, evaluate_model, important_features,feature_selection

def main():
    df_customer = load_data("ACME-HappinessSurvey2020.csv")
    explore_df, tag_to_comment  = explore_data(df_customer)
    X_train, X_test, y_train, y_test = split_data(explore_df, "Y", test_size=0.2)
    best_model, best_params, best_score = tune_hyperparameters(X_train, y_train)
    clf_report = evaluate_model(best_model, X_test, y_test)
    print(clf_report)
    feature_df = important_features(X_train, best_model, tag_to_comment)
    print(feature_df)
    _,_,removed_features,_ = feature_selection(
            X_train, y_train, X_test, y_test, best_params, tag_to_comment
        )  
    print(f"Questions that can be removed from the survey:\n",{tag_to_comment[f] for f in removed_features})  
    print("\n--- Task Done ---")
    sys.exit(0)
    
if __name__ == "__main__":
    main()