import os
from numpy import squeeze
import json
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_and_save_metrics():
    """
    Evaluate the Random Forest Regressor model on the test set and save the predictions and metrics.
    """
    # Load the test set
    X_test = pd.read_csv('data/processed/X_test_scaled.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    y_test = squeeze(y_test, axis=1)

    # Load the model
    model = load('models/rfr_best_model.pkl')

    # Make predictions
    y_pred = model.predict(X_test)

    # Save predictions
    predictions = pd.DataFrame(y_pred, columns=['predicted_silica_concentrate'])
    os.makedirs('data/predicted', exist_ok=True)
    predictions.to_csv('data/predicted/predictions.csv', index=False)

    # Calculate RMSE and R2 score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2: {r2:.4f}")
    metrics = {'MSE': mse, 'R2': r2}

    # Save metrics as json
    os.makedirs('metrics', exist_ok=True)
    with open(os.path.join('metrics','scores.json'), 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    # Check if the model exists
    if os.path.exists('models/rfr_best_model.pkl'):
        evaluate_and_save_metrics()
    else:
        print("Model not found. Please train the model first.")