# Prediction model for an exercise data collection context. Creates a full 5-way
# classification model using scikit-learn. The test data is not labeled, has
# case numbering instead to be used for turning in the predictions and getting a
# # score out of the 20 cases included.

# %% Load packages
#
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split
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
scaler = StandardScaler()
X_df = scaler.fit_transform(train.iloc[:, features])
y = train['label']

# %% Run PCA to transform train into matrix explaining 95% of the variance
#
pca = PCA(n_components=0.95)
pca_fit = pca.fit(X_df)
pca_fit.n_components_
X_pca = pca.transform(X_df)

# %% Fit an SVM model to the transformed matrix, get confusion matrix, accuracy
#
clf = SVC(gamma='auto')
clf_fit = clf.fit(X_pca, y)
clf_fit.score(X_pca, y)
clf._gamma

# %% Run the SVM prediction the 5 validation sets
#
acc = []
for i in range(5):
    fold = validate.sample(20)
    fold_df = scaler.transform(fold.iloc[:, features])
    val = pca.transform(fold_df)
    val_y = fold.loc[:, 'label']
    acc.append(clf_fit.score(val, val_y))
print(acc)
