# Prediction model for an exercise data collection context. Creates a full 5-way
# classification model using SVM. The test data is not labeled, has case
# numbering instead to be used for turning in the predictions and getting a
# score out of the 20 cases included.

# %% Load libraries
#
libraries <- c(
    'caret', 'e1071', 'readr', 'plyr'
)
invisible(lapply(libraries, library, character.only = TRUE, quietly = TRUE))

# %% Load and subset data, eliminate the 100 out of 160 features with no data
#
path <- gsub('\n', '', readr::read_file('.data'))
train_raw <- read.csv(
    file.path(path, 'pml-training.csv'),
    header = TRUE, stringsAsFactors = FALSE,
    colClasses = c(classe = 'factor')
)
train_raw <- rename(train_raw, c('X' = 'id', 'classe' = 'label'))
test_raw <- read.csv(
    file.path(path, 'pml-testing.csv'),
    header = TRUE, stringsAsFactors = FALSE
)
test_raw <- rename(test_raw, c('X' = 'id'))
na_idx <- apply(train_raw, 2, function(x) mean(is.na(x) | x == '') > 0.9)
train_full <- train_raw[, !na_idx]  # 60 features
test <- test_raw[, !na_idx]
features <- 8:59

# %% Set up train dataset and a 20x5 index matrix for 5 validation folds
#
set.seed(78)
train_idx <- caret::createDataPartition(train_full$id, p = 0.95, list = FALSE)
train <- train_full[train_idx, ]
validate <- train_full[-train_idx, ]
val_matrix <- replicate(5, sample(validate$id, 20))

# %% Run PCA to transform train into matrix explaining 95% of the variance
#
pca_fit <- caret::preProcess(train[, features], method = 'pca', thresh = 0.95)
train_pca <- predict(pca_fit, train[, features])

# %% Fit an SVM model to the transformed matrix, get confusion matrix, accuracy
#
svm_fit <- e1071::svm(x = train_pca, y = train$label)
svm_predict <- predict(svm_fit, newdata = train_pca)
conf_matrix <- caret::confusionMatrix(svm_predict, train$label)
print(conf_matrix$table)
print(conf_matrix$overall['Accuracy'])

# %% Run the SVM prediction the 5 validation sets
#
val_test <- function(prep, model, x, y) {
    fold_pca <- predict(prep, x)
    fold_pred <- predict(model, newdata = fold_pca)
    caret::confusionMatrix(fold_pred, y)$overall['Accuracy']
}

acc <- list()
for (i in 1:5) {
    fold <- validate[which(validate$id %in% val_matrix[, i]), ]
    acc[i] <- val_test(pca_fit, svm_fit, fold[, features], fold[, 'label'])
}
print(cbind(acc))

# %% Transform the test sample and run the model to predict final
#
test_pca <- predict(pca_fit, test[, features])
test_pred <- predict(svm_fit, newdata = test_pca)
test_pred
