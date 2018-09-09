# Common functions and variables for version 10 of the Kaggle Credit
# Risk Prediction competition

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import time
from contextlib import contextmanager
import scipy.stats as sstats


# Path to the competition's data files, both .csv and .feather versions
DATA_FILE_FOLDER = "/home/greg/Desktop/Data Projects/Kaggle/Credit Risk Competition/Data Files/"

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print(f"{title} -- done in {time.time() - t0} sec")


def string_col_to_onehot(df, col_name):
    """
    Takes a dataframe and a column name, and creates a one-hot
    dataframe for its values.  This varies slightly from pandas's
    get_dummies function, as it will only return one column if there
    are two unique values in the dataframe, but as many columns as
    unique values if there are more than two.
    """
    dummy_cols = pd.get_dummies(df[col_name], drop_first = False, prefix = col_name)
    if len(dummy_cols.columns) == 2:
        onehot_df = pd.DataFrame({col_name: dummy_cols.iloc[:,0]})
    else:
        onehot_df = dummy_cols
    return onehot_df


def log_average(pd_series):
    """
    Returns the average of the log of a numeric series, exponentiated.
    
    In order to get around the issue of zeros, it adds 1 before taking
    the logarithm and then subtracts 1 at the end.
    """
    return np.exp(np.mean(np.log1p(pd_series))) - 1

def geom_mean(data):
    g_mean = sstats.gmean(np.ma.masked_invalid(data))
    return np.nan if g_mean is np.ma.masked else g_mean


def count_frac_cols(df_gr, col: str, middle_string = ""):
    """
    Takes a grouped dataframe and a column name, and counts
    the number of appearances of each unique string in each group, as
    well as the fraction of the total count each value has.
    
    Includes an optional string to be inserted between the column
    prefix ('NUM' or 'FRAC') and the item name.
    """
    
    group_sizes = df_gr.size()
    
    count_df = df_gr[col].value_counts(dropna = False).unstack(col)
    count_df.drop(np.nan, axis = 1, inplace = True, errors = 'ignore')
    count_df.fillna(0, inplace = True)
    
    item_names = count_df.columns
    if middle_string != "":
        item_names = [middle_string + "_" + item for item in item_names]
    count_names = ["NUM_" + item for item in item_names]
    frac_names = ["FRAC_" + item for item in item_names]
    
    count_df.rename(columns = {i:c for i,c in zip(count_df.columns, count_names)},
                    inplace = True)
    count_df[frac_names] = count_df[count_names].div(group_sizes, axis = 0)
    
    return count_df


def add_polynomial_terms(df, polynomials = {}):
    for col, degree in polynomials.items():
        df[col] = df[col] ** degree
    return df


def log_regress_other_files(file_df, target_df, cols_to_ignore = []):
    """
    Takes the data frame created for a data set (either the train/test
    data or one of the supplemental files) and performs a logistic
    regression on the complete columns in the data frame.
    """
    file_df = file_df.drop(cols_to_ignore, axis = 1, errors = 'ignore').copy()
    
    sc = StandardScaler()
    file_df = file_df.dropna(axis = 1).copy()
    file_df.iloc[:,1:] = sc.fit_transform(file_df.iloc[:,1:])
    
    full_file_data = file_df.drop("SK_ID_CURR", axis = 1).values.astype("float32")
    
    file_df = file_df.loc[file_df["SK_ID_CURR"].isin(target_df["SK_ID_CURR"]),:]
    target = target_df[target_df["SK_ID_CURR"].isin(file_df["SK_ID_CURR"])]["TARGET"]
    
    file_train = file_df.drop("SK_ID_CURR", axis = 1).values.astype("float32")
    
    target = target.values.reshape([len(target),]).astype("float32")
    train_train, train_val, target_train, target_val = train_test_split(file_train, target)
    
    lr = LogisticRegression()
    lr.fit(train_train, target_train)
    
    val_predictions = lr.predict_proba(train_val)
    auc = roc_auc_score(target_val, val_predictions[:,1])
    
    predictions = lr.predict_proba(full_file_data)
    
    return predictions[:,1], auc
