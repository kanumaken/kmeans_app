import pandas as pd

def load_data(file_path):
    """
    Loads and cleans the dataset.
    """
    df = pd.read_csv(file_path)

    # Select relevant numeric columns
    df = df[['TEMP', 'WDSP', 'SLP']].dropna()

    return df
