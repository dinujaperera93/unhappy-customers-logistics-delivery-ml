import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import make_scorer, recall_score, classification_report, accuracy_score
from sklearn.feature_selection import SelectFromModel
from lazypredict.Supervised import LazyClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score


tag_to_comment = {
        "X1": "Order delivered on time",
        "X2": "Contents of the order was as expected",
        "X3": "Ordered everything wanted to order",
        "X4": "Paid a good price for the order",
        "X5": "Satisfied with courier",
        "X6": "The app makes ordering easy"
    }

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def explore_data(df):    
    print(pd.concat([df.head(5), df.tail(5)]))
    print(df.shape)
    
    df.info()
    
    for col in df.loc[:, df.columns != 'Y']:
        plt.figure()
        if col == "Y":
            sns.countplot(data=df, x="Y")
            plt.title('Target Distribution')
        else:
            sns.countplot(data=df, x=col, hue="Y")
            plt.title(tag_to_comment.get(col,col))
   
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()
       
    return df

def split_data(df, target, test_size=0.2):
    X = df.loc[:, df.columns != target]
    y = df[target].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= test_size, random_state=42)
    print(f"Size of the training set: {X_train.shape[0]}\nSize of the testing set: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def select_model(X_train, X_test, y_train, y_test):
    def minority_recall(y_true, y_pred):
        return recall_score(y_true, y_pred, pos_label=0)
    
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=minority_recall)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    print(models)
        
    print(f"\nBest model for minority class: {models.index[0]}")
    return models, predictions
    
def tune_hyperparameters(X_train, y_train):
    
    # Maximise the prediction accuracy of minority class 
    minority_recall = make_scorer(recall_score, pos_label=0)

    space = {
        'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
        'max_depth': hp.choice('max_depth', [3, 5, 7]),
        'min_samples_split': hp.choice('min_samples_split', [5, 10]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [2, 4]),
        'max_features': hp.choice('max_features', ['sqrt', 'log2']),
    }
    
    def objective(params):
        model = ExtraTreesClassifier(**params, random_state=42, class_weight='balanced')
        score = cross_val_score(model, X_train, y_train, cv=10, scoring=minority_recall).mean()
        return {'loss': -score, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
    
    choices = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2'],
    }
    best_params = {k: choices[k][v] for k, v in best.items()}
    
    best_model = ExtraTreesClassifier(**best_params, random_state=42, class_weight='balanced')
    best_model.fit(X_train, y_train)
    best_score = -min(t['result']['loss'] for t in trials.trials)
    
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {round(best_score,2)}")
    
    return best_model, best_params, best_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    clf_report = classification_report(y_test, y_pred)
    return clf_report

def important_features(X_train, model, tag_to_comment):
    importances = model.feature_importances_
    
    feature_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Description': [tag_to_comment[f] for f in X_train.columns],
        'Importance': np.round(importances, 6)
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    return feature_df

def feature_selection(X_train, y_train, X_test, y_test, best_params):
    model_params = {**best_params, 'random_state': 42, 'class_weight': 'balanced'}
    
    clf = Pipeline([
        ('feature_selection', SelectFromModel(ExtraTreesClassifier(**model_params))),
        ('classification', ExtraTreesClassifier(**model_params))
    ])
    
    clf.fit(X_train, y_train)
    
    selected_mask = clf.named_steps['feature_selection'].get_support()
    selected_features = X_train.columns[selected_mask].tolist()
    removed_features = X_train.columns[~selected_mask].tolist()
    
    test_accuracy = round(accuracy_score(y_test, clf.predict(X_test)), 3)
        
    return clf, selected_features, removed_features, test_accuracy
