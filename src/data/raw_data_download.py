import os
import requests


def download_data(url, save_path):
    """Download data from a URL and save it to a specified path."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Data downloaded and saved to {save_path}")
    else:
        print(f"Failed to download data. Status code: {response.status_code}")


if __name__ == "__main__":
    # Define the path to save the data
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw_data')
    print(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, 'raw.csv')

    # Download the raw data from https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv
    # and save it in the data/raw_data directory
    DATA_URL = 'https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv'
    download_data(DATA_URL, save_path)

# The data is now saved in the data/raw_data directory
