import os
os.environ["TQDM_DISABLE"] = "1"

# Patch tqdm notebook
from unittest.mock import MagicMock
import sys
sys.modules['tqdm.notebook'] = MagicMock()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from lazypredict.Supervised import LazyClassifier

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, recall_score, classification_report, accuracy_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from IPython.display import display


# Feature explaination
tag_to_comment = {
    "X1": "Order delivered on time",
    "X2": "Contents of the order was as expected",
    "X3": "Ordered everything wanted to order",
    "X4": "Paid a good price for the order",
    "X5": "Satisfied with courier",
    "X6": "The app makes ordering easy"
}

def load_data(filepath):
    return pd.read_csv(filepath)

def explore_data(df): 
    print(pd.concat([df.head(5), df.tail(5)]))
    print(f"Shape of the dataset : {df.shape}")
    df.info()

def split_data(df, target, seed, test_size=0.2):
    X = df.loc[:, df.columns != target]
    y = df[target].values.ravel()
    
    # Since the data is nearly balanced no stratification is needed, 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= test_size, random_state=seed)
    print(f"Size of the training set: {X_train.shape[0]}\nSize of the testing set: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

# Use EDA only for training data to avoid data leakage
def EDA(X_train, y_train): 
    # Plot the distribution of target and features by the target
    for col in X_train.columns:
        plt.figure()
        sns.countplot(x=X_train[col], hue=y_train)
        plt.title(f"{tag_to_comment.get(col,col)} ({col})")
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()  
    
    plt.figure()
    sns.countplot(x=y_train)
    plt.title('Target Distribution')
    plt.xlabel("Unhappy(0)  Happy (1)")
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

# Tries all 30 models of Lazy Classifier
def select_model(X_train, X_test, y_train, y_test):
    def minority_recall(y_true, y_pred):
        return recall_score(y_true, y_pred, pos_label=0)
    
    # custom_metric=minority_recall, Checks for the improvement of the minority class
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=minority_recall)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    print(models)
    print(f"\nBest model for minority class: {models['minority_recall'].idxmax()}")
    return models, predictions

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    clf_report = classification_report(y_test, y_pred)
    return clf_report

# Get the top four models from lazy classifier
def compare_ensembles(X_train, y_train, X_test, y_test, seed, cv=5):    
    minority_recall = make_scorer(recall_score, pos_label=0)
    
    # The 4 base models from LazyClassifier results
    models = {
        "LGBM": LGBMClassifier(random_state=seed, verbose=-1),
        "Random Forest": RandomForestClassifier(random_state=seed, class_weight="balanced"),
        "ExtraTrees": ExtraTreesClassifier(random_state=seed, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }
    
    base = [(k.lower(), v) for k, v in models.items()]
    models["Voting"] = VotingClassifier(estimators=base, voting="soft")
    models["Stacking"] = StackingClassifier(estimators=base, final_estimator=LogisticRegression(max_iter=2000), cv=5)
    
    results = []
    fitted_models = {}
    
    for name, model in models.items():
        mrec = cross_val_score(model, X_train, y_train, cv=cv, scoring=minority_recall).mean()
        model.fit(X_train, y_train)
        fitted_models[name] = model
        results.append({"Model": name, "Minority_Recall": round(mrec, 4)})
    
    results_df = pd.DataFrame(results).sort_values("Minority_Recall", ascending=False)
   
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=results_df, x="Model", y="Minority_Recall")
    ax.bar_label(ax.containers[0], fmt='%.2f')
    plt.title("Model Comparison - Minority Recall")
    plt.ylabel("Minority Recall")
    plt.grid(True, axis='y')
    plt.show()
    
    return fitted_models, results_df
    
def tune_hyperparameters(X_train, y_train, seed):
    np.random.seed(seed)
    # Maximise the prediction accuracy of minority class 
    minority_recall = make_scorer(recall_score, pos_label=0)

    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 500, 25),
        'max_depth': hp.quniform('max_depth', 2, 8, 1),
        'learning_rate': hp.uniform('learning_rate', 0.05, 0.2),
        'num_leaves': hp.quniform('num_leaves', 5, 31, 5),
        'min_child_samples': hp.quniform('min_child_samples', 10, 30, 5),
        'subsample': hp.uniform('subsample', 0.7, 0.9),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 0.9),
    }
    
    # class_weight='balanced', helps the minority class more
    def objective(params):
        params = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'learning_rate': params['learning_rate'],
            'num_leaves': int(params['num_leaves']),
            'min_child_samples': int(params['min_child_samples']),
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
        }
        model = LGBMClassifier(**params, random_state=seed, verbose=-1, class_weight='balanced')
        # Get the average of 5 scores, with cv=5 , each fold has 20 samples so minaarity class per fold is 8~10 samples
        score = cross_val_score(model, X_train, y_train, cv=5, scoring=minority_recall).mean() 
        return {'loss': -score, 'status': STATUS_OK} # Trial okay
    
    # Objective function, Next combination is picked by Bayesian and try 50 combinations
    trials = Trials()
    rstate = np.random.default_rng(seed) # To get same results for terminal and notebook
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
    
    best_params = {
        'n_estimators': int(best['n_estimators']),
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'num_leaves': int(best['num_leaves']),
        'min_child_samples': int(best['min_child_samples']),
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
    }
    
    best_model = LGBMClassifier(**best_params, random_state=seed, verbose=-1, class_weight='balanced')
    best_model.fit(X_train, y_train)
    best_score = -min(t['result']['loss'] for t in trials.trials)
    
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {round(best_score, 2)}")
    
    return best_model, best_params, best_score

def important_features(X_train, model, tag_to_comment):
    importances = model.feature_importances_
    
    feature_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Description': [tag_to_comment[f] for f in X_train.columns],
        'Importance': np.round(importances, 6)
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    return feature_df

def feature_selection(X_train, y_train, X_test, y_test, best_params, seed):
    model_params = {**best_params, 'random_state': seed, 'class_weight': 'balanced'}
    
    clf = Pipeline([
        ('feature_selection', SelectFromModel(LGBMClassifier(**model_params))),
        ('classification', LGBMClassifier(**model_params))
    ])
    
    clf.fit(X_train, y_train)
    
    selected_mask = clf.named_steps['feature_selection'].get_support()
    selected_features = X_train.columns[selected_mask].tolist()
    removed_features = X_train.columns[~selected_mask].tolist()
    
    test_accuracy = round(accuracy_score(y_test, clf.predict(X_test)), 3)
        
    return clf, selected_features, removed_features, test_accuracy
