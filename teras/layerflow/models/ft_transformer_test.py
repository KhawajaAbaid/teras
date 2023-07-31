from teras.layerflow.models import FTTransformer
from teras.layers.ft_transformer import FTNumericalFeatureEmbedding, FTCLSToken
from teras.layers import CategoricalFeatureEmbedding, Encoder
from teras.utils import get_features_metadata_for_embedding
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os


class FTTransformerTest(tf.test.TestCase):
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
        self.numerical_feature_embedding = FTNumericalFeatureEmbedding(features_metadata=self.features_metadata,
                                                                       embedding_dim=32)
        self.cls_token = FTCLSToken(embedding_dim=32)
        self.encoder = Encoder()
        self.head = keras.layers.Dense(1)
        self.data_batch = data.values

    def test_valid_call(self):
        model = FTTransformer(input_dim=4,
                              categorical_feature_embedding=self.categorical_feature_embedding,
                              numerical_feature_embedding=self.numerical_feature_embedding,
                              cls_token=self.cls_token,
                              encoder=self.encoder,
                              head=self.head)
        model(self.data_batch)

    def test_save_and_load(self):
        model = FTTransformer(input_dim=4,
                              categorical_feature_embedding=self.categorical_feature_embedding,
                              numerical_feature_embedding=self.numerical_feature_embedding,
                              cls_token=self.cls_token,
                              encoder=self.encoder,
                              head=self.head)
        save_path = os.path.join(self.get_temp_dir(), "ft_transformer_lf.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(self.data_batch)
        outputs_reloaded = reloaded_model(self.data_batch)
        self.assertAllClose(outputs_original, outputs_reloaded)
        self.assertAllClose(model.weights, reloaded_model.weights)
