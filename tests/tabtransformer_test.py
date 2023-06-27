from teras.models import TabTransformerClassifier, TabTransformerRegressor
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from teras.utils import get_features_metadata_for_embedding, dataframe_to_tf_dataset
tf.config.run_functions_eagerly(True)

#  <<<<<<<<<<<<<<<<<<<<< REGRESSION Test >>>>>>>>>>>>>>>>>>>>>
print(f"{'-'*15}  REGRESSION TEST {'-'*15}")

# Gemstone dataset from Kaggle's Playground Season 3 Episode 8
# https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
gem_df = pd.read_csv("gemstone_data/train.csv").drop(["id"], axis=1)
cat_cols = ["cut", "color", "clarity"]
num_cols = ["carat", "depth", "table", "x", "y", "z"]

# For FTTransformer, TabTranformer and SAINT, we need to pass a features metadata of dict type
# for features. You can get this dictionary by calling the utility function as below
features_metadata = get_features_metadata_for_embedding(gem_df, categorical_features=cat_cols,
                                                        numerical_features=num_cols)

X_ds = dataframe_to_tf_dataset(gem_df, 'price', batch_size=1024, as_dict=True)

tab_transformer_regressor = TabTransformerRegressor(num_outputs=1,
                                                    features_metadata=features_metadata
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


features_metadata = get_features_metadata_for_embedding(X, cat_cols)

X_ds = dataframe_to_tf_dataset(X, 'income', batch_size=1024, as_dict=True)


tabtransformer_classifier = TabTransformerClassifier(categorical_features_vocabulary=cat_feat_vocab)
tabtransformer_classifier.compile(loss=keras.losses.BinaryCrossentropy(),
                                  optimizer=keras.optimizers.AdamW(learning_rate=0.05),
                                  metrics=["accuracy"])
tabtransformer_classifier.fit(X_ds, epochs=10)
