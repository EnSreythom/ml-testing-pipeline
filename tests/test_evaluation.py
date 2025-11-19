import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_project.evaluation import evaluate_model

def test_perfect_accuracy():
    y_true = [100, 200, 300]
    y_pred = [100, 200, 300]
    result = evaluate_model(y_true, y_pred)
    assert result['rmse'] == 0
    assert result['r2_score'] == 1.0

def test_zero_r2():
    y_true = [100, 200, 300]
    y_pred = [200, 200, 200]
    result = evaluate_model(y_true, y_pred)
    assert result['r2_score'] < 0.1

def test_keys_exist():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    result = evaluate_model(y_true, y_pred)
    assert 'rmse' in result and 'r2_score' in result