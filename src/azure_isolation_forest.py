import mlflow
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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

def split_data(df):
    '''
    splits the data based on normal or anomaly
    '''
    # prepare constants
    labels = df[41]  # for rapid_reusing
    # split by normal vs. anomaly observation
    mask_normal, mask_anomaly = labels=='normal.', labels!='normal.'
    if(DEBUG): print(f'Splitting data with {sum(mask_normal)} NORMAL and {sum(mask_anomaly)} ANOMALY')
    # only select for numerical columns
    df = df.select_dtypes(['number'])
    if(DEBUG): print(f'Data post numerical variable filtering of shape {df.shape}')
    # split data for training (normal)
    X_train, X_test_norm, y_train, y_test_norm = train_test_split(df[mask_normal], labels[mask_normal],
                                                                  shuffle=True, test_size=TEST_SIZE,
                                                                  random_state=RAND_STATE, stratify=labels[mask_normal])
    # split data for testing (anomaly)
    X_test_anom, y_test_anom = df[mask_anomaly], labels[mask_anomaly]
    
    return (X_train, y_train), (X_test_norm, y_test_norm), (X_test_anom, y_test_anom)

def compute_f1(model, data, pos_label):
    '''
    computes an f1 score, pos_label 1=normal, -1=anomaly
    '''
    X, y = data  # unpack data
    true_labels = [pos_label] * X.shape[0]
    predicted_labels = model.predict(X)
    score = f1_score(true_labels, predicted_labels, pos_label=pos_label)
    
    return score

# main method
if __name__ == "__main__":
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='path to the dataset')
    args = parser.parse_args()
    
    # process data
    df = pd.read_csv(args.data_path, index_col=None, header=None)  # read it
    df = process_data(df)

    # split data
    train, test_norm, test_anom = split_data(df)
    X_train, y_train = train  # unpack training data
    
    # train model
    model = IsolationForest(random_state=RAND_STATE)
    model.fit(X_train)
    # TODO: find save model function from sklearn
    # TODO: or turn on mlflow autolog (mlflow.autolog()) <-- check for examples
    
    # score model
    mlflow.log_metric('F1-Score Training Normal', compute_f1(model, train, 1))
    mlflow.log_metric('F1-Score Testing Normal', compute_f1(model, test_norm, 1))
    mlflow.log_metric('F1-Score Testing Anomaly', compute_f1(model, test_anom, -1))

    