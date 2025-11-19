import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_project.pipeline import load_real_data, run_pipeline

def test_real_data_load():
    df = load_real_data("data.csv")
    assert not df.empty
    assert "price" in df.columns

def test_real_pipeline():
    run_pipeline("data.csv", target="price")