from teras.models import TabTransformerClassifier
from tensorflow import keras
from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
from sklearn import datasets as sklearn_datasets
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from teras.utils import get_categorical_features_vocab, dataframe_to_tf_dataset

# (train_data, train_targets), (test_data, test_target) = (boston_housing.load_data())



# Regression Test


# tabnet_regressor = TabNetRegressor(units=64,
#                              decision_step_output_dim=64,
#                              num_decision_steps=5,
#                              relaxation_factor=1.5,
#                              batch_momentum=0.98,
#                              virtual_batch_size=None,
#                              epsilon=1e-5)
#
# tabnet_regressor.compile(loss="MSE", optimizer="Adam", metrics=["MAE"])
# tabnet_regressor.fit(train_data, train_targets, batch_size=32, epochs=10)




# # Classification Test

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

# print("..................Numerical values unique..........")
# for col in num_cols:
#     print(f"{col}: ", X[col].unique())

# print(cat_cols)
# print(num_cols)


# iris_dataset = sklearn_datasets.load_iris()
# X = iris_dataset.data
# y = iris_dataset.target

cat_feat_vocab = get_categorical_features_vocab(X, cat_cols)

# print(cat_feat_vocab)

tabtransformer_classifier = TabTransformerClassifier(categorical_features_vocab=cat_feat_vocab,
                                                     categorical_features=cat_cols,
                                                     numerical_features=num_cols)
tabtransformer_classifier.compile(loss=keras.losses.BinaryCrossentropy(), optimizer="Adam", metrics=["accuracy"])
tabtransformer_classifier.fit(X_ds, epochs=10)