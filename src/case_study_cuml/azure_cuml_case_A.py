import time
import mlflow
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from cuml import RandomForestClassifier as cuRF

'''
this is the carbon intensive methods
'''

# set constants
DEBUG = True
RAND_STATE = 0
TEST_SIZE = 1/4
np.random.seed(RAND_STATE)  # for consistency

# helper functions
def process_data(df):
    '''
    takes raw dataframe and 1) subsets 1% and 2) filters it for top3 labels
    '''
    # subsetting
    if(DEBUG): print(f'Processing {df.head()}')
    if(DEBUG): print(f'Variables {df.columns}')
    if(DEBUG): print(f'Subsetting data of shape {df.shape}')
    subset_size = round(df.shape[0] * 0.01)  # 1% of size
    if(DEBUG): print(f'Subsetting with size {subset_size}')
    valid_idxs = np.random.choice(df.index, size=subset_size, replace=False)
    if(DEBUG): print(f'Subsetting with {subset_size} barcodes')
    mask = df.index.isin(valid_idxs)
    if(DEBUG): print(f'Subsetting with mask with {sum(mask)} positives')
    df = df[mask]
    if(DEBUG): print(f'Subsetted data to shape {df.shape}')
    # filter labels
    labels = df[41]  # get labels, column name is given
    label_counts = labels.value_counts()  # n-obs / label
    mask = label_counts > df.shape[0] * 0.05  # label must match >5% of the data
    valid_labels = label_counts.index[mask]  # get passing labels
    if(DEBUG): print(f'Passing labels = {valid_labels.tolist()}')
    df = df[df[41].isin(valid_labels)]  # subset
    
    return(df)

def encode_data(df, nunique=None):
    '''
    takes in dataframe of categorical variables and one-hot-encodes top @nunique categories with the rest as 'other'
    '''
    # set up one hot encoder
    ohe = OneHotEncoder()
    # set nunique
    if(nunique==None): nunique = 5
    # instantiate the values list
    values_list = []
    # encode columns
    for col in df.columns:
        # get values
        values = df[col].copy()
        # encode values
        counts = values.value_counts()
        ncats = values.nunique()  # get number of categories
        if(ncats <= nunique):  # shrink categories if needed
            valid_cats = counts.index[:nunique]  # get nunique largest categories
            values[~values.isin(valid_cats)] = 'Other'  # create other column
        # add on protocol
        cats = pd.DataFrame(ohe.fit_transform(pd.DataFrame(values)).toarray())
        cats.columns = str(col) + ':' + ohe.categories_[0]
        cats.index = values.index
        # reassign values
        values_list.append(cats)
    # concatenate the one hot encoded values
    values = pd.concat(values_list, axis=1)
    
    return(values)
    
def split_data(df):
    '''
    splits the data based on normal or anomaly
    '''
    # prepare constants
    labels = df[41]  # for rapid reusing
    # split by normal vs. anomaly observation
    mask_normal, mask_anomaly = labels=='normal.', labels!='normal.'
    labels[mask_anomaly] = 'anomaly.'  # concatenate to create binary tree
    if(DEBUG): print(f'Splitting data with {sum(mask_normal)} NORMAL and {sum(mask_anomaly)} ANOMALY')
    # only select for numerical columns
    df_num = df.select_dtypes(['number'])
    if(DEBUG): print(f'Data post numerical variable filtering of shape {df_num.shape}')
    # select for categorical and process them
    df_cat = df.iloc[:,:-1].select_dtypes(['object'])  # ignore label column
    df_cat = encode_data(df_cat)
    if(DEBUG): print(f'Data post categorical variable filtering of shape {df_cat.shape}')
    # combine categorical and numerical data
    df = pd.concat([df_num, df_cat], axis=1).astype('float32')
    if(DEBUG): print(f'Data post encoding with numerical of shape {df.shape}')
    # split data for training (normal)
    X_train, X_test, y_train, y_test = train_test_split(df, labels, shuffle=True, test_size=TEST_SIZE, random_state=RAND_STATE, stratify=labels)
    # split data for training (anomaly)
    mask_normal, mask_anomaly = y_train=='normal.', y_train!='normal.'
    X_train_norm, y_train_norm = X_train[mask_normal], y_train[mask_normal]
    X_train_anom, y_train_anom = X_train[mask_anomaly], y_train[mask_anomaly]
    # split data for testing (anomaly)
    mask_normal, mask_anomaly = y_test=='normal.', y_test!='normal.'
    X_test_norm, y_test_norm = X_test[mask_normal], y_test[mask_normal]
    X_test_anom, y_test_anom = X_test[mask_anomaly], y_test[mask_anomaly]
    
    return (X_train, y_train), (X_train_norm, y_train_norm), (X_train_anom, y_train_anom), (X_test_norm, y_test_norm), (X_test_anom, y_test_anom)

def compute_f1(model, data, pos_label):
    '''
    computes an f1 score, pos_label 1=normal, -1=anomaly
    '''
    label_map = {'normal.': 0, 'anomaly.': 1}  # define constant
    X, y = data  # unpack data
    true_labels = np.vectorize(label_map.get)(y)
    predicted_labels = model.predict(X)
    score = f1_score(true_labels, predicted_labels, pos_label=pos_label)
    
    return score

# main method
if __name__ == "__main__":
    ## PREPROCESS
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='path to the dataset')
    args = parser.parse_args()

    # process data
    df = pd.read_csv(args.data_path, index_col=None, header=None)  # read it
    df = process_data(df)

    # split data
    train, train_norm, train_anom, test_norm, test_anom = split_data(df)
    X_train, y_train = train  # unpack training data

    ## TRAIN
    # get parameters
    label_map = {'normal.': 0, 'anomaly.': 1}
    params = {'random_state':RAND_STATE, 'n_estimators':250, 'max_depth':20,
              'n_bins':20, 'max_samples':1.0, 'max_features':0.4, 'n_streams':1}
    mlflow.log_params(params)
    if(DEBUG): print(f'Random Forest with {params}')
    
    # train model
    model = cuRF(**params)
    start_time = time.time()  # mark start
    model.fit(X_train, np.vectorize(label_map.get)(y_train))

    ## SCORE
    # score model
    f1_train_norm = compute_f1(model, train_norm, 0)
    f1_train_anom = compute_f1(model, train_anom, 1)
    f1_test_norm = compute_f1(model, test_norm, 0)
    f1_test_anom = compute_f1(model, test_anom, 1)

    # get run time
    end_time = time.time()  # mark end
    run_time = end_time - start_time  # calculate runtime based on fitting and scoring

    # log values
    mlflow.log_metric('F1-Score Training Normal', f1_train_norm)
    mlflow.log_metric('F1-Score Training Anomaly', f1_train_anom)
    mlflow.log_metric('F1-Score Testing Normal', f1_test_norm)
    mlflow.log_metric('F1-Score Testing Anomaly', f1_test_anom)
    mlflow.log_metric('Run-Time for Train and Score', run_time)
