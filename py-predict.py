# Prediction model for an exercise data collection context. Creates a full 5-way
# classification model using scikit-learn. The test data is not labeled, has
# case numbering instead to be used for turning in the predictions and getting a
# # score out of the 20 cases included.

# %% Load packages
#
import numpy as np
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold
)
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lib.data_tools import get_data
# from sklearn.linear_model import SGDClassifier

# %% Load the data
#
X, y, validate, test = get_data()

# %% Build pipeline for scaling, pca dimension reduction and the SVC estimator
#
clf = make_pipeline(
    StandardScaler(), PCA(n_components=0.95), SVC(gamma='auto', C=1)
)
clf.fit(X, y)

# %% Use cross-validation to get 5 training scores
#
scores = cross_val_score(
    clf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True)
)
np.around(scores, 3)
for i, s in enumerate(scores):
    print(f'score {i + 1}: {s:.3f}')

# %% Simulate the test with a prediction score for 5 validation sets
#
test_accuracy = []
for i in range(5):
    val = validate.sample(20)
    test_accuracy.append(
        clf.score(val.iloc[:, features], val.loc[:, 'label'])
    )
print(acc)
