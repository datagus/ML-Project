### Analysis of Exercise Quality  
This project builds a prediction model for a barbell lifting task, using a dataset from a 2013 study that captured motion data in several dimensions/calculations for six subjects. The model design combines dimension reduction through PCA, then a support vector machine (SVM) learning process to achieve a 95% in-sample accuracy level for prediction of the manner of exercise among five classifications. Cross-validation testing confirmed the 95% (or greater) accuracy level on three independently sampled validation datasets modeled on the test dataset with an N-size of 20. The model was able to achieve 90% accuracy on the test dataset.

The product includes:  
  - An [R-code file](ex-predict.r) that includes all details of the R code that executes the data analysis pipeline.  
  - A markdown [Final Report](ex-predict-report.md) with select embedded code that describes the analysis work in four main sections:
    * Data cleaning, component understanding and validation dataset sampling
    * Choices made for component reduction and preprocessing using PCA routines
    * Brief description of machine learning model analysis and the choice to use SVM
    * Validation, conclusion, and testing result  

Follow links for the <a href='https://www.coursera.org/learn/practical-machine-learning/supplement/PvInj/course-project-instructions-read-first'>assignment task</a> and the original <a href='https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'>train</a> and <a href='https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'>test</a> datasets.

*Original study and data source credit to:  
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13). Stuttgart, Germany: ACM SIGCHI, 2013.*
