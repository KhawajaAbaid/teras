import tensorflow as tf
from teras.models import FTTransformerClassifier, FTTransformerRegressor
from tensorflow import keras
from tensorflow.keras.datasets import boston_housing
from sklearn import datasets as sklearn_datasets
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from teras.utils import (dataframe_to_tf_dataset,
                         get_features_metadata_for_embedding)
# tf.config.run_functions_eagerly(True)

# There are separate sections for classification and regression tests.


#  <<<<<<<<<<<<<<<<<<<<< REGRESSION Test >>>>>>>>>>>>>>>>>>>>>
print("Running Regression Test...")
# Gemstone dataset from Kaggle's Playground Season 3 Episode 8
# https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
gem_df = pd.read_csv("gemstone_data/train.csv").drop(["id"], axis=1)[:10000]
cat_cols = ["cut", "color", "clarity"]
num_cols = ["carat", "depth", "table", "x", "y", "z"]

# For FTTransformer, TabTranformer and SAINT, we need to pass a dict of features metadata.
# You can get this dict by calling the utility function as below
feature_metadata = get_features_metadata_for_embedding(gem_df,
                                                       categorical_features=cat_cols,
                                                       numerical_features=num_cols)

# If your dataset is heterogenous in that it contains features of numeric and string types,
# then you need the tensorflow dataset in the dict format, otherwise either dict or regular array one works!
X_ds = dataframe_to_tf_dataset(gem_df, 'price', batch_size=1024, as_dict=True)


ft_transformer_regressor = FTTransformerRegressor(features_metadata=feature_metadata)

ft_transformer_regressor.compile(loss="MSE",
                                 optimizer=keras.optimizers.Adam(learning_rate=0.05),
                                 metrics=["MAE"])
ft_transformer_regressor.fit(X_ds, epochs=3)


#  <<<<<<<<<<<<<<<<<<<<< CLASSIFICATION Test >>>>>>>>>>>>>>>>>>>>>
print("\n\nRunning Classification Test...")
adult_df = pd.read_csv("adult_2.csv")[:10000]
X = adult_df
# some preprocessing
X['income'] = LabelEncoder().fit_transform(X['income'])
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
num_cols = ['age', 'hours-per-week']
X.loc[:, cat_cols].replace(to_replace="?", value="missing", inplace=True)
X.loc[:, cat_cols].fillna("missing", inplace=True)
X.loc[:, num_cols].replace(to_replace="?", value=0, inplace=True)
X.loc[:, num_cols].fillna(0, inplace=True)
X[num_cols] = X[num_cols].values.astype(np.float32)


# We need this metadata dictionary
features_metadata = get_features_metadata_for_embedding(X,
                                                        categorical_features=cat_cols,
                                                        numerical_features=num_cols)


# Again, for FTTransformer, TabTransfomer and SAINT, we need to convert our dataframe to tensorflow
# dataset that support retrieving features by indexing column names like X_train["age"] in the model's call method
# And, for that, I have a utility function in teras.utils which is used below.
X_ds = dataframe_to_tf_dataset(X, 'income', batch_size=1024, as_dict=True)

ft_transformer_classifier = FTTransformerClassifier(num_classes=2,
                                                    features_metadata=features_metadata
                                                    )
ft_transformer_classifier.compile(loss=keras.losses.BinaryCrossentropy(),
                                  optimizer=keras.optimizers.Adam(learning_rate=0.05),
                                  metrics=["accuracy"])
ft_transformer_classifier.fit(X_ds, epochs=3)
