import os
os.environ["TQDM_DISABLE"] = "1" # Suppress tqdm progress bars for cleaner output in notebooks and scripts

import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

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

# Mapping of coded feature names to human-readable survey question descriptions.
# Used throughout the pipeline for readable output and report generation.
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
    """Design decision: 80/20 split chosen as a standard ratio for small datasets (126 rows).
    Stratification is not applied because the target classes are nearly balanced (~50/50),
    so random splitting is unlikely to produce a skewed split. A fixed random seed ensures
    reproducibility across runs."""
    X = df.loc[:, df.columns != target]
    y = df[target].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    print(f"Size of the training set: {X_train.shape[0]}\nSize of the testing set: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def EDA(X_train, y_train):
    """Design decision: EDA is performed on training data only to prevent data leakage.
    If test data were included in visualisations, it could influence modelling choices
    (e.g. feature engineering, threshold selection) and produce over-optimistic results."""
    for col in X_train.columns:
        plt.figure()
        sns.countplot(x=X_train[col], hue=y_train)
        plt.title(f"{tag_to_comment.get(col,col)} ({col})")
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(FIGURES_DIR / f"feature_{col}.png", bbox_inches='tight') # Remove white space
        plt.show() 
        plt.close() # Clears this figure from memory
    
    plt.figure()
    sns.countplot(x=y_train)
    plt.title('Target Distribution')
    plt.xlabel("Unhappy(0)  Happy (1)")
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(FIGURES_DIR / "target_distribution.png", bbox_inches='tight') 
    plt.show()
    plt.close()

def select_model(X_train, X_test, y_train, y_test):
    """Design decision: LazyPredict is used as a rapid screening tool to evaluate ~30 classifiers
    in a single pass. This gives a broad baseline view before committing to deeper analysis.

    A custom minority_recall metric (recall with pos_label=0) is used because the business
    objective prioritises identifying unhappy customers. Standard accuracy would mask poor
    performance on the minority class. LazyPredict uses a single train/test split, so these
    results are treated as indicative rather than definitive -- cross-validation follows."""
    def minority_recall(y_true, y_pred):
        return recall_score(y_true, y_pred, pos_label=0)

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=minority_recall)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    print(f"\nBest model for minority class: {models['minority_recall'].idxmax()}")
    return models, predictions

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    clf_report = classification_report(y_test, y_pred)
    return clf_report

def compare_ensembles(X_train, y_train, X_test, y_test, seed, cv=5):
    """Design decision: Four candidate models were selected from the LazyPredict screening
    based on highest minority_recall. These span different learning paradigms:
    - Tree-based: Random Forest, Extra Trees (bagging with balanced class weights)
    - Gradient boosting: LightGBM (sequential correction of errors)
    - Distance-based: KNN (instance-based, no assumptions about data distribution)

    Two ensemble strategies are added on top:
    - Voting (soft): averages predicted probabilities across all base models, which is
      more nuanced than hard voting (majority label). Soft voting benefits from calibrated
      probability estimates.
    - Stacking: uses a LogisticRegression meta-learner to combine base model predictions.
      max_iter=2000 ensures convergence on small datasets.

    5-fold cross-validation provides a more robust estimate than LazyPredict's single split.
    Note: 3 of 4 base models are tree-based, which limits diversity in ensembles. A more
    diverse architecture mix (e.g. SVM, naive Bayes) could improve ensemble generalisation."""
    minority_recall = make_scorer(recall_score, pos_label=0)

    models = {
        "LGBM": LGBMClassifier(random_state=seed, verbose=-1),
        "Random Forest": RandomForestClassifier(random_state=seed, class_weight="balanced"),
        "ExtraTrees": ExtraTreesClassifier(random_state=seed, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }

    base = [(k.lower(), v) for k, v in models.items()]
    # Soft voting averages predicted probabilities rather than taking the majority label
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
    plt.savefig(FIGURES_DIR / "model_comparison.png", bbox_inches='tight')
    plt.show()
    plt.close()
    
    return fitted_models, results_df
    
def tune_hyperparameters(X_train, y_train, seed):
    """Design decision: Hyperopt with TPE (Tree-structured Parzen Estimator) is used instead
    of grid search or random search. 
    
    class_weight='balanced' is set on every candidate model so that the minority class
    (unhappy customers) receives higher weight during training, directly supporting the
    business objective of identifying unhappy customers."""
    np.random.seed(seed)
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

    def objective(params):
        # Hyperopt returns floats (e.g. 100.0) but LightGBM requires int for count parameters
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
        # 5-fold CV: each fold has ~20 samples, so ~8-10 minority class samples per fold
        score = cross_val_score(model, X_train, y_train, cv=5, scoring=minority_recall).mean()
        return {'loss': -score, 'status': STATUS_OK}

    # TPE picks the next combination based on Bayesian modelling of past results
    trials = Trials()
    rstate = np.random.default_rng(seed) # Ensures identical results across notebook and script
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials, rstate=rstate)
    
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
    """Design decision: SelectFromModel with LightGBM is used for model-based feature selection.
    This automatically drops features whose importance falls below the mean importance threshold
    (the sklearn default). A pipeline couples selection and classification so that the same
    tuned hyperparameters are reused consistently. This approach directly answers the business
    question of which survey questions can be removed without harming predictive value."""
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
