import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def test(X_test, y_test, model, model_name):
    y_test_pred = model.predict(X_test)
    results = {
        'model': model_name,
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    return y_test_pred, results
    