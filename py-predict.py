# Prediction model for an exercise data collection context. Creates a full 5-way
# classification model using scikit-learn. The test data is not labeled, has
# case numbering instead to be used for turning in the predictions and getting a
# # score out of the 20 cases included.

# %% Load packages
#
import pandas as pd
# import numpy as np
from os import path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# from sklearn.linear_model import SGDClassifier

# %% Load and subset data, eliminate the 100 out of 160 features with no data
#
with open('.data', 'r') as f:
    data_path = f.readline().replace('\n', '')

train_raw = pd.read_csv(
    path.join(data_path, 'pml-training.csv'),
    dtype={'classe': 'category'}
)
train_raw.rename(
    columns={'Unnamed: 0': 'id', 'classe': 'label'},
    inplace=True
)
test_raw = pd.read_csv(
    path.join(data_path, 'pml-testing.csv'),
    dtype={'classe': 'category'}
)
test_raw.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

na_idx = train_raw.apply(
    lambda x: x.isna().mean() > 0.9, axis=0
)
features = range(7, 59)
train_full = train_raw.loc[:, ~na_idx.values]
test = test_raw.loc[:, ~na_idx.values]

# %% Set up train dataset and a 20x5 index matrix for 5 validation folds
#
train, validate = train_test_split(
    train_full, train_size=0.95, random_state=321
)
X_df = train.iloc[:, features]
y = train['label']

# %% Run PCA to transform train into matrix explaining 95% of the variance
#
pca = PCA(n_components=0.95)
pca_fit = pca.fit(X_df)
pca_fit.n_components_
X_pca = pca.transform(X_df)

# %% Fit an SVM model to the transformed matrix, get confusion matrix, accuracy
#
svm = SVC()
svm_fit = svm.fit(X_pca, y)
svm_predict = svm_fit.predict(X_pca)
np.mean(svm_predict == y)
# %% Run the SVM prediction the 5 validation sets
#
