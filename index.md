# Analysis of Exercise Quality
Gus Mairs  
9/22/2017  

#### Executive Summary
This project builds a machine learning prediction model for a barbell lifting task, using a dataset from a 2013 study[1] that captured motion data in several dimensions/calculations for six subjects. The model design combines principal component reduction (PCA) and a support vector machine training method (SVM) to achieve a 95% in-sample accuracy level for prediction of the manner of exercise among five classifications (A through E). Cross-validation testing confirmed the 95% (or greater) accuracy level on three independently sampled validation datasets modeled on the test dataset with an N-size of 20. The model was able to achieve 90% accuracy on the test dataset.

This paper describes and demonstrates the analysis work completed in four sections:

* Data cleaning, component understanding and validation dataset sampling
* Choices made for component reduction and preprocessing using PCA routines
* Brief description of machine learning model analysis and the choice to use SVM
* Validation, conclusion, and testing result

#### Data cleaning, component understanding and validation sampling
Exploratory analysis of the training and testing datasets quickly revealed the extent of missing or NA values in many of the 152 components columns. After reading the study to understand data collection and calculation techniques, and after examining for potential differences between the training and testing datasets on this point, I determined that removing the components with 97% missing data would improve any model's effectiveness and not cause unintended consequences for the goals of the project.



```r
trnOrig$classe <- as.factor(trnOrig$classe) ## The classe column as a factor
naInd <- apply(trnOrig, 2, function(x) mean(is.na(x) | x == "") > 0.9) ## Index of NA columns
trnBase <- trnOrig[ , !naInd] ## Create a base training dataset
tst <- tstOrig[ , !naInd] ## Final testing dataset
```
With such a large training dataset available, I wanted to enable validation steps for model design by portioning out several reserved full validation sets of 20 cases. These are sampled randomly from the already randomly sampled validation "base" set of 980 cases, and are labeled "vld1" through "vld5".

```r
set.seed(121)
index <- createDataPartition(trnBase$X, p = 0.95, list = FALSE)
trn <- trnBase[index, ] ## Final training set, N = 18,642
valBase <- trnBase[-index, ] ## Validation base set, N = 980
valInd <- replicate(5, sample(valBase$X, 20))
for(i in 1:5) {
        assign(paste0("vld", i), valBase[which(valBase$X %in% valInd[ , i]), ])
        }
```

#### Choices made for component reduction and preprocessing using PCA routines
I started with a hypothesis that a combination of parameter reduction and machine learning algorithm fitting would enable the best efficiency and accuracy in designing a model. I used PCA preprocessing to examine the parameters, choosing thresholds of 80% and 95% of variance explained and getting 12 and 25 components needed, respectively. I ran some initial training efforts with the data transformed by these preprocess models and found decent accuracy and efficiency, so landed on using PCA at the 95% threshold.


```r
(pc95 <- preProcess(trn[ , 8:59], method = "pca", thresh = 0.95))
```

```
## Created from 18642 samples and 52 variables
## 
## Pre-processing:
##   - centered (52)
##   - ignored (0)
##   - principal component signal extraction (52)
##   - scaled (52)
## 
## PCA needed 25 components to capture 95 percent of the variance
```

```r
trnPC95 <- predict(pc95, trn[ , 8:59])
```

#### Machine learning model analysis and the choice to use SVM
I explored model-based (lda), boosting (gbm) and support vector machine (svm) algorithms using the dataset transformed by PCA, with the following outcomes:

* The simple "lda" model worked swiftly but produced only 40-50% in-sample accuracy
* Boosting methods through "gbm" were too computationally intensive on such a large dataset
* Support vector algorithms worked quickly and effectively

I anchored on SVM and was able to produce 95% accuracy without any tuning through an algorithm that completed in under 20 seconds. I noted decent accuracy of 87% with the PC80 transformation -- if the computational overhead was significantly better with just 12 components this might be a good tradeoff, but the PC95 transformation with 25 components runs just as quickly as with 12 components. Here I show the confuion matrix and accuracy result.

```r
library(e1071)
fitSVM95 <- svm(trnPC95, trn$classe)
prdSVM95 <- predict(fitSVM95, newdata = trnPC95)
conf95 <- confusionMatrix(prdSVM95, trn$classe)
conf95$table; conf95$overall["Accuracy"]
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 5283  218    2    6    0
##          B    4 3287   73    4   10
##          C    9   96 3121  270   56
##          D    0    4   49 2772   72
##          E    0    4    5    5 3292
```

```
##  Accuracy 
## 0.9524193
```
#### Validation, conclusion, and testing result
With 95% in-sample accuracy, I want to check for over-fitting and the possibility that my model will not perform well on the test sample. Here I apply the model to each of three of the validation samples to give me a sense of how transferable my model accuracy will be.

```r
vld1PC95 <- predict(pc95, vld1[ , 8:59])
prdVld1PC95 <- predict(fitSVM95, newdata = vld1PC95)
acc.1 <- confusionMatrix(prdVld1PC95, vld1$classe)$overall["Accuracy"]
```


```r
cbind(acc.1, acc.2, acc.3)
```

```
##          acc.1 acc.2 acc.3
## Accuracy  0.95     1  0.95
```
With confidence that I have a sound model for prediction, I ran the model on the test dataset and achieved 90% accuracy. With just one test sample and such a small N-size, this result seems reasonable and close to my validation results. 


[1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13). Stuttgart, Germany: ACM SIGCHI, 2013.
