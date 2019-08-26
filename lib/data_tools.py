# Tools to get the data and clean it up for use in modeling

# %% Load libraries
#
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split

# %% Load and subset data, eliminate the 100 out of 160 features with no data
#
def get_data(seed=123):
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

    na_idx = train_raw.apply(lambda x: x.isna().mean() > 0.9, axis=0)
    features = range(7, 59)
    X_train, X_val, y_train, y_val = train_test_split(
        train_raw.loc[:, ~na_idx.values].iloc[:, features],
        train_raw.loc[:, 'label'],
        train_size=0.95,
        random_state=seed
    )
    test = test_raw.loc[:, ~na_idx.values].iloc[:, features]
    return X_train, y_train, X_val, y_val, test
