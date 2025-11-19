import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from ml_project.data_cleaning import clean_data
from ml_project.normalization import normalize_columns
from ml_project.evaluation import evaluate_model

def load_real_data(path="data.csv"):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Types:\n", df.dtypes)
    print("Sample:\n", df.head())
    return df

def run_pipeline(path="data.csv", target="price"):
    df = load_real_data(path)
    df = clean_data(df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    df = normalize_columns(df, numeric_cols)

    X = df[numeric_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    scores = evaluate_model(y_test, preds)

    print("Evaluation:", scores)