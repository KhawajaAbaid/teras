from tensorflow import keras
from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
from teras.utils import get_categorical_features_vocab
from teras.models import SAINTClassifier

import pandas as pd

# (train_data, train_targets), (test_data, test_target) = (boston_housing.load_data())
#
# train_df = pd.DataFrame(train_data)
#
# # print(train_df.head(5))
#
# categorical_features = [col for col in train_df.columns if train_df[col].nunique() < 50]
# numerical_features = list(set(train_df.columns) - set(categorical_features))
#
# print("Cat cols: ", categorical_features)
#
# cat_vocab = get_categorical_features_vocab(train_df, categorical_features=categorical_features)
#
# print(numerical_features)
#
# saint_model = SAINT(categorical_features=categorical_features,
#                     numerical_features=numerical_features,
#                     categorical_features_vocab=cat_vocab,
#
# )
#
# saint_model.compile(loss="MSE", optimizer="Adam", metrics=["MAE"])
#
# saint_model.fit(train_df, train_targets, batch_size=32, epochs=10)


# CLASSIFICATION TEST
from sklearn.preprocessing import LabelEncoder
from teras.utils import dataframe_to_tf_dataset
import numpy as np

adult_df = pd.read_csv("X:\\Khawaja Abaid\\Self Learning try\\Datasets\\adult_2.csv")
X = adult_df
X['income'] = LabelEncoder().fit_transform(X['income'])

cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
num_cols = ['age', 'hours-per-week']

X.loc[:, cat_cols].replace(to_replace="?", value="missing", inplace=True)
X.loc[:, cat_cols].fillna("missing", inplace=True)
#
X.loc[:, num_cols].replace(to_replace="?", value=0, inplace=True)
X.loc[:, num_cols].fillna(0, inplace=True)
#
X[num_cols] = X[num_cols].values.astype(np.float32)

# print(X.head())

X_ds = dataframe_to_tf_dataset(X, 'income', batch_size=512)

cat_feat_vocab = get_categorical_features_vocab(X, cat_cols)

# print(cat_feat_vocab)

saint_classifier = SAINTClassifier(num_classes=2,
                                    categorical_features_vocab=cat_feat_vocab,
                                    categorical_features=cat_cols,
                                    numerical_features=num_cols)
saint_classifier.compile(loss=keras.losses.BinaryCrossentropy(), optimizer="Adam", metrics=["accuracy"])
saint_classifier.fit(X_ds, epochs=10)