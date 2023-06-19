import keras.losses

from teras.models.tabnet import TabNetClassifier, TabNetRegressor
import tensorflow as tf
from sklearn import datasets as sklearn_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd
import numpy as np
from teras.utils.utils import get_categorical_features_vocabulary, dataframe_to_tf_dataset
tf.config.run_functions_eagerly(True)


#  <<<<<<<<<<<<<<<<<<<<< REGRESSION Test >>>>>>>>>>>>>>>>>>>>>
print(f"{'-'*15}  REGRESSION TEST {'-'*15}")

# Gemstone dataset from Kaggle's Playground Season 3 Episode 8
# https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
gem_df = pd.read_csv("gemstone_data/train.csv").drop(["id"], axis=1)
cat_cols = ["cut", "color", "clarity"]
num_cols = ["carat", "depth", "table", "x", "y", "z"]
# gem_df = gem_df.drop(cat_cols, axis=1)

# TODO: WE NEEEEEED TO CONVERT CATEGORICAL VALUES TO INTEGERS (ORDINALLY ENCODE THEM)

oe = OrdinalEncoder()
gem_df[cat_cols] = oe.fit_transform(gem_df[cat_cols])

cat_feat_vocab = get_categorical_features_vocabulary(gem_df, cat_cols, key="idx")

training_df, pretrain_df = train_test_split(gem_df, test_size=0.25, shuffle=True, random_state=1337)

X_ds = dataframe_to_tf_dataset(training_df, 'price', batch_size=1024, as_dict=False)
pretrain_ds = dataframe_to_tf_dataset(pretrain_df, batch_size=1024, as_dict=False)

tabnet_regressor = TabNetRegressor(categorical_features_vocabulary=cat_feat_vocab)

# Configure Pretrainer's fit() method arguments
tabnet_regressor.pretrainer_fit_config.epochs = 2
# Call pretrain
tabnet_regressor.pretrain(pretrain_ds, num_features=gem_df.shape[1])
# Train the regressor for our main task
tabnet_regressor.compile(loss="mse", metrics=["mae"])
tabnet_regressor.fit(X_ds, epochs=3)
