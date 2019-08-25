# Prediction model for an exercise data collection context. Creates a full 5-way
# classification model using scikit-learn. The test data is not labeled, has
# case numbering instead to be used for turning in the predictions and getting a
# # score out of the 20 cases included.

# %% Load packages
#
import pandas as pd
from os import path
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# from sklearn.linear_model import SGDClassifier

# %% Load and subset data, eliminate the 100 out of 160 features with no data
#
with open('.data', 'r') as f:
    data_path = f.readline().replace('\n', '')

train_raw = pd.read_csv(
    path.join(data_path, 'pml-training.csv'),
    dtype={'classe': 'category'},
    low_memory=False
)
train_raw.rename(
    columns={'Unnamed: 0': 'id', 'classe': 'label'},
    inplace=True
)
test_raw = pd.read_csv(
    path.join(data_path, 'pml-testing.csv'),
    dtype={'classe': 'category'},
    low_memory=False
)
test_raw.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

na_idx = train_raw.apply(lambda x: x.isna().mean() > 0.9, axis=0)
train_full = train_raw.loc[:, ~na_idx.values]
test = test_raw.loc[:, ~na_idx.values]
features = range(7, 59)

# %% Set up and scale train dataset and validation set for 5 folds later
#
train, validate = train_test_split(
    train_full, train_size=0.95, random_state=123
)
# scaler = StandardScaler()
# X_df = scaler.fit_transform(train_full.iloc[:, features])
X_df = train.iloc[:, features]
y = train['label']

# %% Build pipeline for scaling, pca dimension reduction and the SVC estimator
#
clf = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),
    SVC(gamma='auto', C=1)
)
clf.fit(X_df, y)

# %% Use cross-validation to get 5 training scores
#
scores = cross_val_score(
    clf, X_df, y, cv=StratifiedKFold(n_splits=5, shuffle=True)
)
scores

# %% Run the SVM prediction the 5 validation sets
#
acc = []
for i in range(5):
    val = validate.sample(20)
    acc.append(
        clf.score(val.iloc[:, features], val.loc[:, 'label'])
    )
print(acc)
