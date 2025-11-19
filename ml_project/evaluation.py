from sklearn.metrics import r2_score

from sklearn.metrics import root_mean_squared_error

def evaluate_model(y_true, y_pred):
    return {
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred)
    }