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
    return df

def split_data(df):
    """
    Splits the data into features and target
    and into training and testing sets.
    """

    # Our target variable is 'silica_concentrate'
    target = df['silica_concentrate']
    features = df.drop(columns=['silica_concentrate'])

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
