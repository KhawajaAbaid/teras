from teras.models.tabnet import TabNetClassifier, TabNetRegressor, TabNet, TabNetPretrainer
import tensorflow as tf
from sklearn import datasets as sklearn_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd
import numpy as np
from teras.utils.utils import get_features_metadata_for_embedding, dataframe_to_tf_dataset
tf.config.run_functions_eagerly(True)


#  <<<<<<<<<<<<<<<<<<<<< REGRESSION Test >>>>>>>>>>>>>>>>>>>>>
print(f"{'-'*15}  REGRESSION TEST {'-'*15}")

# Gemstone dataset from Kaggle's Playground Season 3 Episode 8
# https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
gem_df = pd.read_csv("gemstone_data/train.csv").drop(["id"], axis=1)[:1600]
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

features_metadata = get_features_metadata_for_embedding(gem_df,
                                                        categorical_features=cat_cols,
                                                        numerical_features=num_cols)


training_df, pretrain_df = train_test_split(gem_df, test_size=0.25, shuffle=True, random_state=1337)

pretrain_df.pop("price")
X_ds = dataframe_to_tf_dataset(training_df, 'price', batch_size=1024, as_dict=True)
pretrain_ds = dataframe_to_tf_dataset(pretrain_df, batch_size=1024, as_dict=True)

# NEW DISCOVERY: If the categorical values have been encoded, you MUST set the encode_categorical_values param to False
# other otherwise it the string lookup layer will throw an error.

tabnet = TabNet(features_metadata=features_metadata,
                          encode_categorical_values=False,
                          virtual_batch_size=4)

tabnet_pretrainer = TabNetPretrainer(model=tabnet)
tabnet_pretrainer.compile()
print("pretraining...")
tabnet_pretrainer.fit(pretrain_ds, epochs=3)

# Retrieve the pretrained instance
pretrained_tabent = tabnet_pretrainer.pretrained_model

# Create a tabnet regressor instance based off the pretrained model
# to finetune on the main task at hand
tabnet_regressor = TabNetRegressor.from_pretrained(pretrained_model=pretrained_tabent,
                                                   num_outputs=1)

# The returned instance is not compiled -- for obvious reasons to allow user the flexibility
# freeze compile train, unfreeze compile train, you know the typical fine-tuning workflow.

# First we'll train the head and keep the base freezed
print("Training the top/head layer...")
pretrained_tabent.trainable = False
tabnet_regressor.compile(loss="mse", metrics=["mae"])
tabnet_regressor.fit(X_ds, epochs=3)

# Then we'll unfreeze the base and train the whole model
print("Finetuning the whole model together...")
pretrained_tabent.trainable = True
tabnet_regressor.compile(loss="mse", metrics=["mae"])
tabnet_regressor.fit(X_ds, epochs=3)
print()
