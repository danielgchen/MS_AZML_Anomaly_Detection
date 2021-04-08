# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to
import pandas as pd

# FIND VALID LABELS
def get_valid_labels(values, perc = None):
    '''
    picks the top labels that encompass {perc} of the data
    '''
    # set default values
    if (perc is None):
        perc = 0.05  # five percent of the data
    # subset labels and count observations
    label_counts = values.value_counts()  # get n-obs / label
    mask = label_counts > values.shape[0] * perc  # create label mask
    valid_labels = label_counts.index[mask]  # subset labels
    # report status
    print(f'Passing labels = {valid_labels.tolist()}')
    # return data
    return valid_labels


# The entry point function MUST have two input arguments.
# If the input port is not connected, the corresponding
# dataframe argument will be None.
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1 = None, dataframe2 = None):
    # Execution logic goes here

    # subset the data with the top labels
    raw_labels = dataframe1.iloc[:, -1]  # get label column - # is empirically determined
    valid_labels = get_valid_labels(raw_labels)  # find top labels
    dataframe1 = dataframe1.loc[raw_labels.isin(valid_labels)]  # subset data
    del raw_labels, valid_labels  # clean up

    # split up the data for training (1) and testing (2)
    mask_train = dataframe1.iloc[:, -1] == 'normal.'  # train (normal)
    mask_test = dataframe1.iloc[:, -1] != 'normal.'  # test (anomalies)
    data = dataframe1.iloc[:, :-1].select_dtypes(['number'])  # grab data
    print(data.head())
    print(f'n-train{sum(mask_train)}')
    print(f'n-test{sum(mask_test)}')
    data_train = data.loc[mask_train, :]  # construct train dataframe
    data_test = data.loc[mask_test, :]  # construct test dataframe

    # Return value must be of a sequence of pandas.DataFrame
    # E.g.
    #   -  Single return value: return dataframe1,
    #   -  Two return values: return dataframe1, dataframe2
    return data_train, data_test
