Task 13: Model Optimization

Objective
Optimize a classifier using hyperparameter tuning via GridSearchCV and RandomizedSearchCV, tuning n_estimators and max_depth with cross-validation.

Tools
Python, Scikit-learn, NumPy, SciPy, Joblib


Dataset
Breast Cancer Wisconsin dataset (built into scikit-learn, load_breast_cancer) — 569 samples, 30 features, binary classification (malignant/benign).

Approach
Loaded dataset and split into train (80%) / test (20%), stratified by class.
Trained a baseline RandomForestClassifier with default parameters.
GridSearchCV: exhaustive search over
      n_estimators: [50, 100, 150, 200]
      max_depth: [None, 5, 10, 15, 20]
     5-fold cross-validation, scoring = accuracy
RandomizedSearchCV: 20 random combinations sampled from
    n_estimators ~ randint(50, 300)
    max_depth ~ randint(2, 30)
    5-fold cross-validation, scoring = accuracy
Compared both methods and selected the best model based on CV accuracy.
Evaluated final model on test set with classification report.
Saved the optimized model and a best-parameters report.


Results

Method              Best Params                            CV Accuracy                   Test Accuracy            Time
Baseline             default                               -                             0.9561                  -
GridSearchCV         n_estimators=150, max_depth=None      0.9604                        0.9561                  ~25s
RandomizedSearchCV   n_estimators=229, max_depth=8         0.9604                        0.9561                  ~36s
