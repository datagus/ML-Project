# Prediction model for an exercise data collection context. Creates a full 5-way
# classification model using scikit-learn. The test data is not labeled, has
# case numbering instead to be used for turning in the predictions and getting a
# # score out of the 20 cases included.

# %% Load packages
#
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
X, y, X_val, y_val, test = get_data(seed=90)

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
for i, s in enumerate(scores):
    print(f'score {i + 1}: {s:.3f}')

# %% Simulate the final test for small samples of size 20
#
test_accuracy = []
for i in range(5):
    vx = X_val.sample(20)
    vy = y_val.loc[vx.index]
    test_accuracy.append(
        sum(clf.predict(vx) == vy)
    )
for i, s in enumerate(test_accuracy):
    print(f'score {i + 1}: {s}')
