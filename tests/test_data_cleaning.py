import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from ml_project.data_cleaning import clean_data

def test_row_reduction():
    df = pd.DataFrame({'A': [1, 1, 2], 'B': [3, 3, None]})
    cleaned = clean_data(df)
    assert cleaned.shape[0] == 1