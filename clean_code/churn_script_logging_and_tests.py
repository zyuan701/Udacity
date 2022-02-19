# docstring
"""
Predict Customer Churn Rate - Test File

Author: Z.Y.
Date: Feb 12, 2022

"""
import os
import logging
import math
import churn_library as clib


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = clib.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    df = clib.import_data("./data/bank_data.csv")
    try:
        clib.perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: The file wasn't found")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    df = clib.import_data("./data/bank_data.csv")
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    try:
        df_encoded = clib.encoder_helper(df, category_lst)
        logging.info("Categorical Column Encoded: SUCCESS")
    except TypeError as err:
        logging.error("Testing encoder_helper: ERROR")
        raise err

    try:
        assert df_encoded.shape[1] - df.shape[1] == len(category_lst)
        logging.info("Number of new columns matches")
    except AssertionError as err:
        logging.error("Number of new columns mismatches")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df = clib.import_data("./data/bank_data.csv")
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df_encoded = clib.encoder_helper(df, category_lst)
    column_kept_lst = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    try:
        (_, x_test, _, _) = clib.perform_feature_engineering(
            df_encoded, column_kept_lst)
        logging.info("Perform Feature Engineering: SUCCESS")
        assert x_test.shape[0] == math.ceil(df.shape[0] * 0.3)
        logging.info("Train test set split: SUCCESS")
    except AssertionError as err:
        logging.error("Train test set split size mismatch")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    df = clib.import_data("./data/bank_data.csv")
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df_encoded = clib.encoder_helper(df, category_lst)
    column_kept_lst = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    X_train, X_test, y_train, y_test = clib.perform_feature_engineering(
        df_encoded, column_kept_lst)
    try:
        clib.train_models(X_train, X_test, y_train, y_test)
        logging.info("model training: SUCCESS")
        assert os.path.isfile("./models/logistic_model.pkl") is True
        logging.info('logistic_model.pkl is found')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    try:
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info('rfc_model.pkl is found')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    try:
        assert os.path.isfile("./images/results/roc_curve_models.png") is True
        logging.info('roc_curve_models.png is found')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
