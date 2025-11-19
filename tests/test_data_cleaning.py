import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from ml_project.data_cleaning import clean_data

def test_remove_duplicates():
    df = pd.DataFrame({'A': [1, 1, 2], 'B': [3, 3, 4]})
    cleaned = clean_data(df)
    assert cleaned.shape[0] == 2

def test_drop_nulls():
    df = pd.DataFrame({'A': [1, None, 2], 'B': [3, 4, None]})
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0

def test_row_reduction():
    df = pd.DataFrame({'A': [1, 1, None], 'B': [3, 3, 4]})
    cleaned = clean_data(df)
    assert cleaned.shape[0] < df.shape[0]