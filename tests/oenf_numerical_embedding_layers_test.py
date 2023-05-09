from tensorflow import keras
import tensorflow as tf
import pandas as pd
from teras.layers import PeriodicEmbedding, TabTransformerCategoricalFeatureEmbedding
from teras.utils import get_categorical_features_vocab, dataframe_to_tf_dataset
from teras.preprocessing import PiecewiseLinearEncoding
# tf.config.run_functions_eagerly(True)


# Here we'll test numerical embedding layers proposed by
# Yury et al. in the paper On Embeddings for Numerical Features

# The PeriodicEmbedding layer is a layer so it is used within the Regressor model
# On the other hand, the PiecewiseLinearEncoding is an encoding class like
# those encoding classes in sklearn, so it used in the data preprocessing step


class Regressor(keras.Model):
    def __init__(self, embedding_dim=32,
                 categorical_features=None,
                 numerical_features=None,
                 categorical_features_vocab=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.categorical_features = categorical_features
        self.numerical_features = tuple(numerical_features)
        self.categorical_features_vocab = categorical_features_vocab

        self.categorical_feature_embedding = TabTransformerCategoricalFeatureEmbedding(
                                                        categorical_features=self.categorical_features,
                                                        categorical_features_vocab=self.categorical_features_vocab,
                                                        embedding_dim=self.embedding_dim
                                                        )
        # Period Embedding layer to embed numerical features
        self.periodic_embedding = PeriodicEmbedding(embedding_dim=self.embedding_dim,
                                                    n_features=len(self.numerical_features),
                                                    sigma=0.01)

        self.concat = keras.layers.Concatenate(axis=1)

        self.dense_hidden = keras.layers.Dense(16, activation="relu")
        self.dense_out = keras.layers.Dense(1)

    def call(self, inputs):
        cat_embeds = self.categorical_feature_embedding(inputs)
        numerical_data = tf.transpose(([inputs[feature] for feature in self.numerical_features]))
        num_embeds = self.periodic_embedding(numerical_data)
        x = self.concat([cat_embeds, num_embeds])
        x = self.dense_hidden(x)
        out = self.dense_out(x)
        return out

# <<<<<<<<<<<<<<<<<<< REGRESSION Test >>>>>>>>>>>>>>>>>>>>>>>>>>
print(f"{'-' * 15}  REGRESSION TEST {'-' * 15}")

# Gemstone dataset from Kaggle's Playground Season 3 Episode 8
# https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
gem_df = pd.read_csv("gemstone_data/train.csv").drop(["id"], axis=1)
cat_cols = ["cut", "color", "clarity"]
num_cols = ["carat", "depth", "table", "x", "y", "z"]

# =====> PREPROCESSING <======

# The Piecewise Linear Encoding class is implemented like encoding classes
# in the sklearn library - it doesn't subclass layers.Layer,
# hence the reason it's put in preprocessing module instead of layers

ple = PiecewiseLinearEncoding(task="regression", method="tree")
ple.fit(gem_df[num_cols], y=gem_df["price"])
gem_df[num_cols] = ple.transform(gem_df[num_cols])

cat_vocab = get_categorical_features_vocab(gem_df, cat_cols)

gem_ds = dataframe_to_tf_dataset(gem_df, target="price")


# =====> TRAINING THE MODEL <======

regressor_model = Regressor(categorical_features=cat_cols,
                            numerical_features=num_cols,
                            categorical_features_vocab=cat_vocab)

regressor_model.compile(loss="MSE",
                      optimizer=keras.optimizers.AdamW(learning_rate=0.01),
                      metrics=["MAE"])
regressor_model.fit(gem_ds, epochs=10)
