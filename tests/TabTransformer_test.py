from teras.models import TabTransformerClassifier, TabTransformerRegressor
from tensorflow import keras
import tensorflow as tf
from sklearn import datasets as sklearn_datasets
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from teras.utils import get_categorical_features_vocab, dataframe_to_tf_dataset

#  <<<<<<<<<<<<<<<<<<<<< REGRESSION Test >>>>>>>>>>>>>>>>>>>>>
print(f"{'-'*15}  REGRESSION TEST {'-'*15}")

# Gemstone dataset from Kaggle's Playground Season 3 Episode 8
# https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
gem_df = pd.read_csv("gemstone_data/train.csv").drop(["id"], axis=1)
cat_cols = ["cut", "color", "clarity"]
num_cols = ["carat", "depth", "table", "x", "y", "z"]

# print(cat_cols)
# print(num_cols)

# For FTTransformer, TabTranformer and SAINT, we need to pass a vacobulary of dict type
# for the categorical features. You can get this vocab by calling the utility function as below
cat_feat_vocab = get_categorical_features_vocab(gem_df, cat_cols)
# print(cat_feat_vocab)

# For FTTransformer, TabTransfomer and SAINT, we need to convert our dataframe to tensorflow
# dataset that support retrieving features by indexing column names like X_train["age"] in the model's call method
# And, for that, I have a utility function in teras.utils which is used below.
X_ds = dataframe_to_tf_dataset(gem_df, 'price', batch_size=1024)

tab_transformer_regressor = TabTransformerRegressor(units_out=1,
                                                  numerical_features=num_cols,
                                                  categorical_features=cat_cols,
                                                  categorical_features_vocab=cat_feat_vocab
                                                    )
tab_transformer_regressor.compile(loss="MSE",
                                  optimizer=keras.optimizers.AdamW(learning_rate=0.05),
                                  metrics=["MAE"])
tab_transformer_regressor.fit(X_ds, epochs=10)




# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Classficiation Test >>>>>>>>>>>>>>>>>>>>>>>
print(f"\n\n{'-'*15}  CLASSIFICATION TEST {'-'*15}")
adult_df = pd.read_csv("adult_2.csv")
X = adult_df
X['income'] = LabelEncoder().fit_transform(X['income'])

cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
num_cols = ['age', 'hours-per-week']

# some typical preprocessing
X.loc[:, cat_cols].replace(to_replace="?", value="missing", inplace=True)
X.loc[:, cat_cols].fillna("missing", inplace=True)
X.loc[:, num_cols].replace(to_replace="?", value=0, inplace=True)
X.loc[:, num_cols].fillna(0, inplace=True)
X[num_cols] = X[num_cols].values.astype(np.float32)

# For FTTransformer, TabTranformer and SAINT, we need to pass a vacobulary of dict type
# for the categorical features. You can get this vocab by calling the utility function as below
cat_feat_vocab = get_categorical_features_vocab(X, cat_cols)

# For FTTransformer, TabTransfomer and SAINT, we need to convert our dataframe to tensorflow
# dataset that support retrieving features by indexing column names like X_train["age"] in the model's call method
# And, for that, I have a utility function in teras.utils which is used below.
X_ds = dataframe_to_tf_dataset(X, 'income', batch_size=1024)


tabtransformer_classifier = TabTransformerClassifier(categorical_features_vocab=cat_feat_vocab,
                                                     categorical_features=cat_cols,
                                                     numerical_features=num_cols)
tabtransformer_classifier.compile(loss=keras.losses.BinaryCrossentropy(),
                                  optimizer=keras.optimizers.AdamW(learning_rate=0.05),
                                  metrics=["accuracy"])
tabtransformer_classifier.fit(X_ds, epochs=10)