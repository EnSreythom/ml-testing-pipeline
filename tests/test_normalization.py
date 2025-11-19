import pandas as pd
import pytest
from ml_project.normalization import normalize_column

def test_normalized_range():
    df = pd.DataFrame({'score': [10, 20, 30]})
    norm = normalize_column(df, 'score')
    assert norm.min() >= 0 and norm.max() <= 1

def test_length_preserved():
    df = pd.DataFrame({'score': [10, 20, 30]})
    norm = normalize_column(df, 'score')
    assert len(norm) == len(df)

def test_invalid_column():
    df = pd.DataFrame({'score': [10, 20, 30]})
    with pytest.raises(KeyError):
        normalize_column(df, 'missing')