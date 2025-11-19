import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd 
from ml_project.normalization import normalize_columns

import pandas as pd
from ml_project.normalization import normalize_columns

def test_normalize_columns():
    df = pd.DataFrame({'score': [10, 20, 30]})
    df = normalize_columns(df, ['score'])
    assert df['score'].min() == 0
    assert df['score'].max() == 1