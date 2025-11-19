import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data():
    """Loads a simple dataset."""
    return pd.DataFrame({'feature': [0, 1, 1, 0], 'target': [0, 1, 1, 0]})

def train_model(X, y):
    """Trains a logistic regression model."""
    model = LogisticRegression()
    model.fit(X, y)
    return model

def evaluate_model(y_true, y_pred):
    """Returns accuracy score."""
    return {'accuracy': accuracy_score(y_true, y_pred)}