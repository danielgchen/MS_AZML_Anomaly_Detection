# The script MUST define a class named AzureMLModel.
# This class MUST at least define the following three methods: "__init__", "train" and "predict".
# The signatures (method and argument names) of all these methods MUST be exactly the same as the following example.

# Please do not install extra packages such as "pip install xgboost" in this script,
# otherwise errors will be raised when reading models in down-stream modules.

import pandas as pd
from sklearn.ensemble import IsolationForest


class AzureMLModel:
    # The __init__ method is only invoked in module "Create Python Model",
    # and will not be invoked again in the following modules "Train Model" and "Score Model".
    # The attributes defined in the __init__ method are preserved and usable in the train and predict method.
    def __init__(self):
        # self.model must be assigned
        self.model = IsolationForest(random_state = 0)  # set for consistency
        self.feature_column_names = list()

    # Train model
    #   Param<df_train>: a pandas.DataFrame
    #   Param<df_label>: a pandas.Series
    def train(self, df_train, df_label):
        # self.feature_column_names records the column names used for training.
        # It is recommended to set this attribute before training so that the
        # feature columns used in predict and train methods have the same names.
        self.feature_column_names = df_train.columns.tolist()
        self.model.fit(df_train)

    # Predict results and retrieves the anomaly score
    #   Param<df>: a pandas.DataFrame
    #   Must return a pandas.DataFrame
    def predict(self, df):
        # The feature columns used for prediction MUST have the same names as the ones for training.
        # The name of score column ("Scored Labels" in this case) MUST be different from any other
        # columns in input data.
        return pd.DataFrame({'Scored Labels': self.model.predict(df[self.feature_column_names]),
                             'Scored Anomaly Value': self.model.decision_function(df[self.feature_column_names])})
