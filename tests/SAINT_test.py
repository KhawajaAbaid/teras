from tensorflow import keras
from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from teras.utils import get_categorical_features_vocab, dataframe_to_tf_dataset
from teras.models import SAINTClassifier, SAINTRegressor


#  <<<<<<<<<<<<<<<<<<<<< REGRESSION Test >>>>>>>>>>>>>>>>>>>>>
print(f"{'-'*15} REGRESSION TEST {'-'*15}")
# Gemstone dataset from Kaggle's Playground Season 3 Episode 8
# https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
gem_df = pd.read_csv("gemstone_data/train.csv").drop(["id"], axis=1)
cat_cols = ["cut", "color", "clarity"]
num_cols = ["carat", "depth", "table", "x", "y", "z"]


# For FTTransformer, TabTranformer and SAINT, we need to pass a vacobulary of dict type
# for the categorical features. You can get this vocab by calling the utility function as below
cat_feat_vocab = get_categorical_features_vocab(gem_df, cat_cols)
# print(cat_feat_vocab)

# For FTTransformer, TabTransfomer and SAINT, we need to convert our dataframe to tensorflow
# dataset that support retrieving features by indexing column names like X_train["age"] in the model's call method
# And, for that, I have a utility function in teras.utils which is used below.
X_ds = dataframe_to_tf_dataset(gem_df, 'price', batch_size=512)


saint_regressor = SAINTRegressor(units_out=1,
                                          numerical_features=num_cols,
                                          categorical_features=cat_cols,
                                          categorical_features_vocab=cat_feat_vocab
                                         )

saint_regressor.compile(loss="MSE", optimizer="Adam", metrics=["MAE"])
saint_regressor.fit(X_ds, batch_size=32, epochs=10)




# <<<<<<<<<<<<<<<<<<<<<<<<<<< CLASSIFICATION TEST >>>>>>>>>>>>>>>>>>>>>>>
print(f"{'-'*15} Classification TEST {'-'*15}")

adult_df = pd.read_csv("adult_2.csv")
X = adult_df

# some data preprocessing
X['income'] = LabelEncoder().fit_transform(X['income'])
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
num_cols = ['age', 'hours-per-week']
X.loc[:, cat_cols].replace(to_replace="?", value="missing", inplace=True)
X.loc[:, cat_cols].fillna("missing", inplace=True)
X.loc[:, num_cols].replace(to_replace="?", value=0, inplace=True)
X.loc[:, num_cols].fillna(0, inplace=True)
X[num_cols] = X[num_cols].values.astype(np.float32)


# For FTTransformer, TabTranformer and SAINT, we need to pass a vacobulary of dict type
# for the categorical features. You can get this vocab by calling the utility function as below
X_ds = dataframe_to_tf_dataset(X, 'income', batch_size=512)

# For FTTransformer, TabTransfomer and SAINT, we need to convert our dataframe to tensorflow
# dataset that support retrieving features by indexing column names like X_train["age"] in the model's call method
# And, for that, I have a utility function in teras.utils which is used below.
cat_feat_vocab = get_categorical_features_vocab(X, cat_cols)

# print(cat_feat_vocab)

saint_classifier = SAINTClassifier(num_classes=2,
                                    categorical_features_vocab=cat_feat_vocab,
                                    categorical_features=cat_cols,
                                    numerical_features=num_cols)
saint_classifier.compile(loss=keras.losses.BinaryCrossentropy(), optimizer="Adam", metrics=["accuracy"])
saint_classifier.fit(X_ds, epochs=10)