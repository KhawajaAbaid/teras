from teras.layerflow.models import (TabTransformer,
                                    TabTransformerClassifier,
                                    TabTransformerRegressor,
                                    TabTransformerPretrainer)
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from teras.utils import get_features_metadata_for_embedding, dataframe_to_tf_dataset
tf.config.run_functions_eagerly(True)

#  <<<<<<<<<<<<<<<<<<<<< REGRESSION Test >>>>>>>>>>>>>>>>>>>>>
print(f"{'-'*15}  REGRESSION TEST {'-'*15}")

# Gemstone dataset from Kaggle's Playground Season 3 Episode 8
# https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
gem_df = pd.read_csv("gemstone_data/train.csv").drop(["id"], axis=1)[:20480]
cat_cols = ["cut", "color", "clarity"]
num_cols = ["carat", "depth", "table", "x", "y", "z"]

# For FTTransformer, TabTranformer and SAINT, we need to pass a features metadata of dict type
# for features. You can get this dictionary by calling the utility function as below
features_metadata = get_features_metadata_for_embedding(gem_df, categorical_features=cat_cols,
                                                        numerical_features=num_cols)

pretraining_ds = gem_df.copy()
pretraining_ds.pop("price")
pretraining_ds = dataframe_to_tf_dataset(pretraining_ds, batch_size=4096, as_dict=True)
X_ds = dataframe_to_tf_dataset(gem_df, 'price', batch_size=4096, as_dict=True)


# Use a base TabTransformer instance for pretraining
tab_transformer = TabTransformer(features_metadata=features_metadata)
tab_transformer_pretrainer = TabTransformerPretrainer(model=tab_transformer)
# We have default loss and optimizer values in place already - though feel free to use your own
tab_transformer_pretrainer.compile()
tab_transformer_pretrainer.fit(pretraining_ds, epochs=1)
# Retrieve the pretrained base model
pretrained_model = tab_transformer_pretrainer.pretrained_model
# Use to base pretrained model to build a regressor model
tabnet_regressor = TabTransformerRegressor.from_pretrained(pretrained_model=pretrained_model)
tabnet_regressor.compile(loss="mse", metrics=["mae"], optimizer=keras.optimizers.AdamW(learning_rate=0.05))
tabnet_regressor.fit(X_ds, epochs=3)