from tensorflow import keras
from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from teras.utils import get_features_metadata_for_embedding, dataframe_to_tf_dataset
from teras.models import SAINTClassifier, SAINTRegressor
from teras.models.saint import SAINTPretrainer
# tf.config.run_functions_eagerly(True)


#  <<<<<<<<<<<<<<<<<<<<< REGRESSION Test >>>>>>>>>>>>>>>>>>>>>
print(f"{'-'*15} REGRESSION TEST {'-'*15}")
# Gemstone dataset from Kaggle's Playground Season 3 Episode 8
# https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
gem_df = pd.read_csv("gemstone_data/train.csv").drop(["id"], axis=1)
cat_cols = ["cut", "color", "clarity"]
num_cols = ["carat", "depth", "table", "x", "y", "z"]


# For FTTransformer, TabTranformer and SAINT, we need to pass features metadata of dict type
# You can get this metadata by calling the utility function as below
features_metadata = get_features_metadata_for_embedding(gem_df, cat_cols, num_cols)

X_ds = dataframe_to_tf_dataset(gem_df, 'price', batch_size=1024, as_dict=True)

pretrain_df = gem_df.copy()
pretrain_df.pop("price")
X_pretrain = dataframe_to_tf_dataset(pretrain_df, as_dict=True)

saint_regressor = SAINTRegressor(num_outputs=1,
                                 features_metadata=features_metadata
                                )

pretrainer = SAINTPretrainer(model=saint_regressor)
pretrainer.compile(optimizer=keras.optimizers.AdamW(learning_rate=0.05))
pretrainer.fit(X_ds, epochs=3)

saint_regressor.compile(loss="MSE",
                        optimizer=keras.optimizers.AdamW(learning_rate=0.05),
                        metrics=["MAE"])
saint_regressor.fit(X_ds, epochs=10)


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


X_ds = dataframe_to_tf_dataset(X, 'income', batch_size=1024, as_dict=True)

features_metadata = features_metadata(X, cat_cols)

saint_classifier = SAINTClassifier(num_classes=2, features_metadata=features_metadata)
saint_classifier.compile(loss=keras.losses.BinaryCrossentropy(),
                         optimizer=keras.optimizers.AdamW(learning_rate=0.05),
                         metrics=["accuracy"])
saint_classifier.fit(X_ds, epochs=10)