# Tools to get the data and clean it up for use in modeling

# %% Load libraries
#
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split

# %% Load and subset data, eliminate the 100 out of 160 features with no data
#
def get_data():
    with open('.data', 'r') as f:
        data_path = f.readline().replace('\n', '')

    train_raw = pd.read_csv(
        path.join(data_path, 'pml-training.csv'),
        dtype={'classe': 'category'},
        low_memory=False
    ).rename(columns={'Unnamed: 0': 'id', 'classe': 'label'})

    test_raw = pd.read_csv(
        path.join(data_path, 'pml-testing.csv'),
        dtype={'classe': 'category'},
        low_memory=False
    ).rename(columns={'Unnamed: 0': 'id'})

    features = range(7, 59)
    na_idx = train_raw.apply(lambda x: x.isna().mean() > 0.9, axis=0)
    train = train_raw.loc[:, ~na_idx.values].iloc[:, features]
    train_y = train_raw.loc[:, 'label']
    test = test_raw.loc[:, ~na_idx.values].iloc[:, features]
    train, val = train_test_split(
        train, train_size=0.95, random_state=123
    )
    return train, train_y, val, test
