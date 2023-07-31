import tensorflow as tf
from tensorflow import keras
from teras.layerflow.models.tabtransformer import (TabTransformer,
                                                   TabTransformerPretrainer)
from teras.layers import TabTransformerColumnEmbedding
from teras.layers import CategoricalFeatureEmbedding, Encoder
from teras.utils import get_features_metadata_for_embedding
import pandas as pd
import numpy as np
import os


class TabTransformerTest(tf.test.TestCase):
    def setUp(self):
        data = pd.DataFrame({"income": np.ones(10),
                             "goals": np.ones(10),
                             'player_level': [5, 7, 9, 8, 9, 10, 9, 7, 8, 9],
                             'shirt_number': [7, 10, 10, 7, 7, 10, 10, 10, 7, 10]})
        categorical_feats = ["player_level", "shirt_number"]
        numerical_feats = ["income", "goals"]
        self.features_metadata = get_features_metadata_for_embedding(pd.DataFrame(data),
                                                                     categorical_features=categorical_feats,
                                                                     numerical_features=numerical_feats)
        self.categorical_feature_embedding = CategoricalFeatureEmbedding(features_metadata=self.features_metadata,
                                                                         embedding_dim=32)
        self.numerical_feature_normalization = keras.layers.LayerNormalization()
        self.column_embedding = TabTransformerColumnEmbedding(num_categorical_features=2)
        self.encoder = Encoder()
        self.head = keras.layers.Dense(1)
        self.data_batch = data.values

    def test_valid_call(self):
        model = TabTransformer(input_dim=4,
                               categorical_feature_embedding=self.categorical_feature_embedding,
                               column_embedding=self.column_embedding,
                               numerical_feature_normalization=self.numerical_feature_normalization,
                               encoder=self.encoder,
                               head=self.head)
        model(self.data_batch)

    def test_save_and_load(self):
        model = TabTransformer(input_dim=4,
                               categorical_feature_embedding=self.categorical_feature_embedding,
                               column_embedding=self.column_embedding,
                               numerical_feature_normalization=self.numerical_feature_normalization,
                               encoder=self.encoder,
                               head=self.head)
        save_path = os.path.join(self.get_temp_dir(), "tabtransformer_lf.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(self.data_batch)
        outputs_reloaded = reloaded_model(self.data_batch)
        self.assertAllClose(outputs_original, outputs_reloaded)


class TabTransformerPretrainerTest(tf.test.TestCase):
    def setUp(self):
        data = pd.DataFrame({"income": np.ones(10),
                             "goals": np.ones(10),
                             'player_level': [5, 7, 9, 8, 9, 10, 9, 7, 8, 9],
                             'shirt_number': [7, 10, 10, 7, 7, 10, 10, 10, 7, 10]})
        categorical_feats = ["player_level", "shirt_number"]
        numerical_feats = ["income", "goals"]
        self.features_metadata = get_features_metadata_for_embedding(pd.DataFrame(data),
                                                                     categorical_features=categorical_feats,
                                                                     numerical_features=numerical_feats)
        self.categorical_feature_embedding = CategoricalFeatureEmbedding(features_metadata=self.features_metadata,
                                                                         embedding_dim=32)
        self.numerical_feature_normalization = keras.layers.LayerNormalization()
        self.column_embedding = TabTransformerColumnEmbedding(num_categorical_features=2)
        self.encoder = Encoder()
        self.data_batch = data.values
        self.num_features = 4
        self.num_features_to_replace = 2
        feature_indices_to_replace = tf.random.uniform((tf.shape(self.data_batch)[0], self.num_features_to_replace),
                                                       maxval=self.num_features,
                                                       dtype=tf.int32)
        self.mask = tf.reduce_max(tf.one_hot(feature_indices_to_replace,
                                             depth=self.num_features),
                                  axis=1)

    def test_valid_call(self):
        base_model = TabTransformer(input_dim=4,
                                    categorical_feature_embedding=self.categorical_feature_embedding,
                                    column_embedding=self.column_embedding,
                                    numerical_feature_normalization=self.numerical_feature_normalization,
                                    encoder=self.encoder)
        pretrainer = TabTransformerPretrainer(model=base_model,
                                              features_metadata=self.features_metadata)
        pretrainer(self.data_batch, mask=self.mask)

    def test_save_and_load(self):
        base_model = TabTransformer(input_dim=4,
                                    categorical_feature_embedding=self.categorical_feature_embedding,
                                    column_embedding=self.column_embedding,
                                    numerical_feature_normalization=self.numerical_feature_normalization,
                                    encoder=self.encoder)
        pretrainer = TabTransformerPretrainer(model=base_model,
                                              features_metadata=self.features_metadata)
        save_path = os.path.join(self.get_temp_dir(), "tabtransformer_pretrainer_lf.keras")
        pretrainer.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = pretrainer(self.data_batch, mask=self.mask)
        outputs_reloaded = reloaded_model(self.data_batch, mask=self.mask)
        # We can't check for AllClose because the call method randomly shuffles the inputs
