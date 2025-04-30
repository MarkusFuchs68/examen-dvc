import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(file_path):
    """
    Reads the data from a CSV file, splits it into features and target,
    and then splits it into training and testing sets and saves them to CSV files
    into data/processed_data.
    """
    df = pd.read_csv(file_path)

    # Our target variable is 'silica_concentrate'
    target = df['silica_concentrate']
    features = df.drop(columns=['date','silica_concentrate'])

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Save the training and testing sets to CSV files
    dest_data_folder = os.path.join(os.getcwd(), 'data', 'processed')
    os.makedirs(dest_data_folder, exist_ok=True)
    X_train.to_csv(os.path.join(dest_data_folder,'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(dest_data_folder,'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(dest_data_folder,'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(dest_data_folder,'y_test.csv'), index=False)

    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    """
    Standardizes the features in the training and testing sets.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaled data to CSV files
    dest_data_folder = os.path.join(os.getcwd(), 'data', 'processed')
    pd.DataFrame(X_train_scaled).to_csv(os.path.join(dest_data_folder,'X_train_scaled.csv'), index=False)
    pd.DataFrame(X_test_scaled).to_csv(os.path.join(dest_data_folder,'X_test_scaled.csv'), index=False)

    return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    # Example usage
    file_path = os.path.join(os.getcwd(), 'data', 'raw', 'raw.csv')
    X_train, X_test, y_train, y_test = split_data(file_path)
    standardize_data(X_train, X_test)
    print("Data has been split, standardized and saved to data/processed.")