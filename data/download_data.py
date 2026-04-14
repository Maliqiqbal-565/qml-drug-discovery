import pandas as pd
import os


def download_delaney_dataset(data_path='data/raw/delaney-processed.csv'):
    """Download the Delaney dataset if not already present."""
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    if not os.path.exists(data_path):
        print(f'Downloading dataset to {data_path}...')
        df = pd.read_csv(url)
        df.to_csv(data_path, index=False)
        print('Download complete.')
    else:
        print(f'Dataset already exists at {data_path}')

    return pd.read_csv(data_path)


if __name__ == '__main__':
    download_delaney_dataset()
