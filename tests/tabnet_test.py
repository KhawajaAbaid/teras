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

# The architecture can handle both encoded and not encoded (even string) values

# If some categorical features contain string values, then create the tensorflow dataset by passing as_dict True.
# Otherwise, it's up to you to create a dictionary format dataset or regular array format one.

# If categorical values have been encoded, set the encode_categorical_values to False, otherwise set it to True.

# Let's try it out!

oe = OrdinalEncoder()
gem_df[cat_cols] = oe.fit_transform(gem_df[cat_cols])

cat_feat_vocab = get_categorical_features_vocabulary(gem_df, cat_cols)

training_df, pretrain_df = train_test_split(gem_df, test_size=0.25, shuffle=True, random_state=1337)

X_ds = dataframe_to_tf_dataset(training_df, 'price', batch_size=1024, as_dict=False)
pretrain_ds = dataframe_to_tf_dataset(pretrain_df, batch_size=1024, as_dict=False)

# NEW DISCOVERY: If the categorical values have been encoded, you MUST set the encode_categorical_values param to False
# other otherwise it the string lookup layer will throw an error.

tabnet_regressor = TabNetRegressor(categorical_features_vocabulary=cat_feat_vocab,
                                   encode_categorical_values=False)

# Configure Pretrainer's fit() method arguments
tabnet_regressor.pretrainer_fit_config.epochs = 2
# Call pretrain
# tabnet_regressor.pretrain(pretrain_ds, num_features=gem_df.shape[1])
# Train the regressor for our main task
tabnet_regressor.compile(loss="mse", metrics=["mae"])
tabnet_regressor.fit(X_ds, epochs=3)
