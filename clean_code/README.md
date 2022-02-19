# Predict Customer Churn

- The project aims to predict the churn rate for a credit card customer based on their characteristics on the database.

## Project Description
This project will go through 4 parts in total.
1. EDA
- dataset distribution interm of churn, customer age, marital staus, transaction counts and correlation among all characteristics.

2. feature engineering
- categorical data transformation
+ turn each categorical column into a new column with proportion of churn for each category

- train test set split
+ 70% as the train set, 30% of the original dataset as test.

3. model training 
- algorithm: logistic regression, random forest

4. model evaluation
- feature importance
- evaluation plots: roc_curve
- model performance report


## Running Files
- pip install scikit-learn==0.22.2
- ipython churn_library.py python_script_logging_and_tests.py
- autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
- autopep8 --in-place --aggressive --aggressive churn_library.py
- pylint churn_library.py
- pylint churn_script_logging_and_tests.py
