# %%
library(caret)

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
index <- createDataPartition(trn_base$X, p = 0.95, list = FALSE)
train <- trn_base[index, ] ## Final training set, n = 18,642
val_set <- trn_base[-index, ] ## Validation base set, n = 980
val_idx <- replicate(5, sample(val_set$X, 20))
for (i in 1:5) {
    assign(
        paste0('vld', i),
        val_set[which(val_set$X %in% val_idx[, i]), ]
    )
}

# %% Run PCA to capture features explaining 95% of the variance
#
pca_95 <- caret::preProcess(train[, 8:59], method = 'pca', thresh = 0.95)
pca_95
pred_pca_95 <- predict(pca_95, train[, 8:59])

# %%
library(e1071)
fitSVM95 <- svm(pred_pca_95, train$classe)
prdSVM95 <- predict(fitSVM95, newdata = pred_pca_95)
conf95 <- confusionMatrix(prdSVM95, train$classe)
conf95$table; conf95$overall['Accuracy']

# %%
vld1PC95 <- predict(pca_95, vld1[ , 8:59])
prdVld1PC95 <- predict(fitSVM95, newdata = vld1PC95)
acc.1 <- confusionMatrix(prdVld1PC95, vld1$classe)$overall['Accuracy']

# %%
vld2PC95 <- predict(pca_95, vld2[ , 8:59])
prdVld2PC95 <- predict(fitSVM95, newdata = vld2PC95)
acc.2 <- confusionMatrix(prdVld2PC95, vld2$classe)$overall['Accuracy']

vld3PC95 <- predict(pca_95, vld3[ , 8:59])
prdVld3PC95 <- predict(fitSVM95, newdata = vld3PC95)
acc.3 <- confusionMatrix(prdVld3PC95, vld3$classe)$overall['Accuracy']

# %%
cbind(acc.1, acc.2, acc.3)

# %%
## Transform the test sample and run the model to prodict for the quiz
tstPC95 <- predict(pca_95, tst[ , 8:59])
prdTstPC95 <- predict(fitSVM95, newdata = tstPC95)
prdTstPC95
