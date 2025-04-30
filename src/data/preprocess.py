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
    features = df.drop(columns=['silica_concentrate'])

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

if __name__ == "__main__":
    # Example usage
    file_path = os.path.join(os.getcwd(), 'data', 'raw', 'raw.csv')
    split_data(file_path)
    print("Data has been split and saved to data/processed.")