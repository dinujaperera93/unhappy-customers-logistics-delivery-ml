import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def explore_data(df):
    
    tag_to_comment = {
        "X1": "Order delivered on time",
        "X2": "Contents of the order was as expected",
        "X3": "Ordered everything wanted to order",
        "X4": "Paid a good price for the order",
        "X5": "Satisfied with courier",
        "X6": "The app makes ordering easy"
    }
    
    print(pd.concat([df.head(5), df.tail(5)]))
    print(df.shape)
    
    df.info()
    
    for col in df.loc[:, df.columns != 'Y']:
        plt.figure()
        sns.countplot(data=df, x=col, hue="Y")
        plt.title(f'Distribution of {col}')
        plt.xlabel(f'{tag_to_comment[col]}')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()
        
    plt.figure()
    sns.countplot(data=df, x="Y")
    plt.title('Distribution of Y')
    plt.xlabel('Happiness')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show() 
       
    return df

def split_data(df, target, test_size=0.2):
    X = df.loc[:, df.columns != target]
    y = df.loc[:, df.columns == target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size= test_size, random_state=42)
    print(f"Size of the training set: {X_train.shape[0]}\nSize of the testing set: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test
    
def tune_hyperparameters(X_train, y_train):
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [1, 3, 6, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train.values.ravel())
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score}")
    
    return best_model, best_params, best_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    clf_report = classification_report(y_test, y_pred)
    return clf_report

def important_features(model):
    best_features = model.feature_importances_
    return best_features
