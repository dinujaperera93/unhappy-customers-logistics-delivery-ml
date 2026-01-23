import sys
from customer_model import load_data, explore_data,split_data, tune_hyperparameters, evaluate_model, important_features

def main():
    df_customer = load_data("ACME-HappinessSurvey2020.csv")
    explore_df = explore_data(df_customer)
    X_train, X_test, y_train, y_test = split_data(df_customer, "Y", test_size=0.2)
    best_model, best_params, best_score = tune_hyperparameters(X_train, y_train)
    clf_report = evaluate_model(best_model, X_test, y_test)
    print(clf_report)
    best_features = important_features(best_model)
    print(best_features)
    print("\n--- Done ---")
    sys.exit(0)
    
if __name__ == "__main__":
    main()