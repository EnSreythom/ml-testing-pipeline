import pandas as pd

def clean_data(df):
    """Removes duplicates and null values from a DataFrame."""
    return df.drop_duplicates().dropna()