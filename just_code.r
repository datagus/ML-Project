# %%
library(caret, quietly = T)
library(e1071)

# %% Load, subset and adjust the data
#
trn_u <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
tst_u <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
trn_raw <- read.csv(trn_u, header = TRUE, stringsAsFactors = FALSE)
tst_raw <- read.csv(tst_u, header = TRUE, stringsAsFactors = FALSE)

trn_raw$classe <- as.factor(trn_raw$classe)
na_idx <- apply(
    trn_raw, 2, function(x) mean(is.na(x) | x == '') > 0.9
)
trn_base <- trn_raw[, !na_idx] ## Create a base training dataset
tst <- tst_raw[, !na_idx] ## Final testing dataset

# %% Set up train, 5 validate folds, and test data sets
#
set.seed(121)
index <- caret::createDataPartition(trn_base$X, p = 0.95, list = FALSE)
train <- trn_base[index, ] ## Final training set, n = 18,642
val_set <- trn_base[-index, ] ## Validation base set, n = 980
val_idx <- replicate(5, sample(val_set$X, 20))
for (i in 1:5) {
    assign(
        paste0('cv_', i),
        val_set[which(val_set$X %in% val_idx[, i]), ]
    )
}

# %% Run PCA to transform train explaining 95% of the variance
#
pca_fit <- caret::preProcess(train[, 8:59], method = 'pca', thresh = 0.95)
trn_pca <- predict(pca_fit, train[, 8:59])

# %% Fit an SVM model to the transformed matrix, confusion matrix, accuracy
#
svm_fit <- e1071::svm(trn_pca, train$classe)
svm_pred <- predict(svm_fit, newdata = trn_pca)
trn_cm <- caret::confusionMatrix(svm_pred, train$classe)
print(trn_cm$table)
print(trn_cm$overall['Accuracy'])

# %% Run the SVM prediction on 3 of the validation sets
#
cv_1_pca <- predict(pca_fit, cv_1[, 8:59])
cv_1_pred <- predict(svm_fit, newdata = cv_1_pca)
acc_1 <- confusionMatrix(cv_1_pred, cv_1$classe)$overall['Accuracy']

cv_2_pca <- predict(pca_fit, cv_2[, 8:59])
cv_2_pred <- predict(svm_fit, newdata = cv_2_pca)
acc_2 <- confusionMatrix(cv_2_pred, cv_2$classe)$overall['Accuracy']

cv_3_pca <- predict(pca_fit, cv_3[, 8:59])
cv_3_pred <- predict(svm_fit, newdata = cv_3_pca)
acc_3 <- confusionMatrix(cv_3_pred, cv_3$classe)$overall['Accuracy']

cbind(acc_1, acc_2, acc_3)

# %% Transform the test sample and run the model to predict final
#
tst_pca <- predict(pca_fit, tst[, 8:59])
tst_pred <- predict(svm_fit, newdata = tst_pca)
tst_cm <- caret::confusionMatrix(tst_pred, tst$classe)
print(tst_cm$table)
print(tst_cm$overall['Accuracy'])
tst_pred
