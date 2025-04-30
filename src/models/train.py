import os
from numpy import squeeze
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from joblib import dump


def train_and_save_model():
    """
    Trains a Random Forest Regressor on the preprocessed data and saves the best model.
    """
    # Load our preprocessed, scaled data
    data_path = os.path.join(os.getcwd(), 'data', 'processed')
    X_train = pd.read_csv(os.path.join(data_path, 'X_train_scaled.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test_scaled.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
    y_train = squeeze(y_train, axis=1)  # Convert DataFrame to Series
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
    y_test = squeeze(y_test, axis=1)  # Convert DataFrame to Series

    # Define model and parameter grid
    model = RandomForestRegressor(random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    # GridSearch with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test MSE: {mse:.4f}")

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save the best model
    dump(best_model, os.path.join('models', 'rfr_best_model.pkl'))


if __name__ == "__main__":
    # Example usage
    train_and_save_model()
    print("Model has been trained, and model saved to models/rfr_best_model.pkl.")