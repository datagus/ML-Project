## Porting to Python  

Fully ported the project from R over to Python with 4 goals:

#### Replicate the original result  
- Both `e1071` in R and `scikit-learn` use the `libsvm` C-code libraries at the foundation of calculations -- once basic modeling parameters matched across platforms the prediction results were almost identical
- PCA implementation (targeting components that explain 0.95 of the variance) are just slightly different - choosing 26 components in `caret` versus 25 in `sklearn-svm`  

#### Explore algorithm speed differences  
  - The `scikit-learn` algorithm runs about 5x faster and makes it far easier to cross-validate and tune. I did not dig in (yet) to try to learn why the speed difference would be so great  

####  Compare coding approaches and efficiency  
  - Writing in Python encouraged more use of chunking the tools into a [library of functions](pycode/lib/data_tools.py) and notebook-like modeling flow  
  - Creates efficiencies in the better (more explicit) object-orientation to both data pipeline operations and machine learning modeling  
  - Could reverse-port to refine the original exploratory R code around these ideals  

#### Learn how the ML packages operate across platforms  
  - The platform difference is minor in essential terms and fairly easy to manage translation of the modeling moves, but Python's clearer object-orientation to the tools and coding process makes writing readable code easier  
  - Both `caret` and `e-1071` produce objects to contain model attributes, but seem to use function notation to extract or operate on objects and attributes. Some further work to refine R code in the Pythonic way would help clarify these differences
  - Python better to put this modeling into production for a team  
  - If a paper is the goal using R-markdown (perhaps in RStudio) may the better workflow to embed code chunks and output and produce a PDF

#### Further tuning hyperparameters  
  - Running the polynomial kernel for the SVM model improved accuracy by just a little over the default radial basis kernel  
  - Using the linear kernel was far worse  
  - Whitening the PCA transformation made the predictions worse  
  - Brief tuning of the 'gamma' constant and 'C' cost hyperparameter did not improve the result over the default (gamma = 1/n_parameters and C = 1)  
  - Applied SVM learning process using gradient descent learning process via `SGDClassifier` and it performed very poorly

kernel | pca | pca whitened
:---: | :----------: | :--------:
rbf | 0.96 | 0.95
poly | 0.97 | 0.92  
