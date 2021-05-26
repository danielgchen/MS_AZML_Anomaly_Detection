import time
import mlflow
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
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
    if(DEBUG): print(f'Splitting data with {sum(mask_normal)} NORMAL and {sum(mask_anomaly)} ANOMALY')
    # only select for numerical columns
    df_num = df.select_dtypes(['number'])
    if(DEBUG): print(f'Data post numerical variable filtering of shape {df_num.shape}')
    # select for categorical and process them
    df_cat = df.iloc[:,:-1].select_dtypes(['object'])  # ignore label column
    df_cat = encode_data(df_cat)
    if(DEBUG): print(f'Data post categorical variable filtering of shape {df_cat.shape}')
    # combine categorical and numerical data
    df = pd.concat([df_num, df_cat], axis=1)
    if(DEBUG): print(f'Data post encoding with numerical of shape {df.shape}')
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

    # score data
    score_df = pd.DataFrame(columns=['n_estimators','max_samples','max_features','run_time',
                                     'f1_train_norm','f1_test_norm','f1_test_anom'])
    for n_estimators in range(50,501,50):  # 10 or 50 jumps
        for max_samples in range(64,257,64):  # 8, 16, 32, or 64 jumps
            for max_features in range(25,101,25):  # 5, 10, 25 jumps
                # train model
                model = IsolationForest(random_state=RAND_STATE, n_estimators=n_estimators,
                                        max_samples=max_samples, max_features=max_features/100)
                start_time = time.time()  # mark start
                model.fit(X_train)
                end_time = time.time()  # mark end
                run_time = end_time - start_time  # calculate runtime

                # score model
                f1_train_norm = compute_f1(model, train, 1)
                f1_test_norm = compute_f1(model, test_norm, 1)
                f1_test_anom = compute_f1(model, test_anom, -1)

                # save metrics
                idx = score_df.shape[0]
                print(f'P{idx}, n-estimators: {n_estimators}, max-samples: {max_samples}, max-features: {max_features}')
                score_df.loc[idx] = [n_estimators,max_samples,max_features,run_time,
                                     f1_train_norm,f1_test_norm,f1_test_anom]
    # log scoring dataframe
    mlflow.log_dict(score_df.to_dict(), 'score_df.json')
    
    # plot metrics
    ylabel_map = {'f1_train_norm':'F1-Score Training Normal', 'f1_test_norm':'F1-Score Testing Normal',
                  'f1_test_anom':'F1-Score Testing Anomaly', 'run_time': 'Run Time (Seconds)'}
    fig = plt.figure(figsize=[12,16])
    idx = 1  # counter for subplots
    for metric in ['f1_train_norm','f1_test_norm','f1_test_anom','run_time']:
        for param in ['n_estimators','max_samples','max_features']:
            # setup figure
            ax = fig.add_subplot(4, 3, idx)
            xs,ys = score_df[param], score_df[metric]
            # scatter
            ax.scatter(xs, ys)
            # fit line
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            sample_xs = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(sample_xs, p(sample_xs), linestyle='--', color='k')
            # label axes
            ax.set(xlabel=param.replace('_',' ').title(), ylabel=ylabel_map[metric])
            # prepare for next round
            idx += 1
    fig.tight_layout()
    mlflow.log_figure(fig, 'param_vs_metric.png')

    # visualization via pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(score_df.iloc[:,:3].T)  # we're only using the parameters
    # - per param
    fig = plt.figure(figsize=[12,4])
    idx = 1
    for param in ['n_estimators','max_samples','max_features']:
        ax = fig.add_subplot(1, 3, idx)
        ax.scatter(pca.components_[0], pca.components_[1], c=score_df[param])
        ax.set(xlabel='PC1', ylabel='PC2', title=param)
        idx += 1
    fig.tight_layout()
    mlflow.log_figure(fig, 'pca_params.png')
    
    # per metric
    fig = plt.figure(figsize=[16,4])
    idx = 1
    for metric in ['f1_train_norm','f1_test_norm','f1_test_anom','run_time']:
        ax = fig.add_subplot(1, 4, idx)
        ax.scatter(pca.components_[0], pca.components_[1], c=score_df[metric])
        ax.set(xlabel='PC1', ylabel='PC2', title=metric)
        idx += 1
    fig.tight_layout()
    mlflow.log_figure(fig, 'pca_metrics.png')