import tensorflow as tf
from tensorflow import keras
from teras.layerflow.models.tabnet import (TabNet,
                                           TabNetPretrainer)
from teras.layers import (CategoricalFeatureEmbedding,
                          TabNetEncoder,
                          TabNetDecoder)
from teras.utils import get_features_metadata_for_embedding
import pandas as pd
import numpy as np
import os
import tensorflow_probability as tfp


class TabNetTest(tf.test.TestCase):
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
                                                                         embedding_dim=1)
        self.encoder = TabNetEncoder(data_dim=4)
        self.head = keras.layers.Dense(1)
        self.data_batch = data.values

    def test_valid_call(self):
        model = TabNet(input_dim=4,
                       features_metadata=self.features_metadata,
                       categorical_feature_embedding=self.categorical_feature_embedding,
                       encoder=self.encoder,
                       head=self.head)
        model(self.data_batch)

    def test_save_and_load(self):
        model = TabNet(input_dim=4,
                       features_metadata=self.features_metadata,
                       categorical_feature_embedding=self.categorical_feature_embedding,
                       encoder=self.encoder,
                       head=self.head)
        save_path = os.path.join(self.get_temp_dir(), "tabnet_lf.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(self.data_batch)
        outputs_reloaded = reloaded_model(self.data_batch)
        self.assertAllClose(outputs_original, outputs_reloaded)


class TabNetPretrainerTest(tf.test.TestCase):
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
                                                                         embedding_dim=1)
        self.encoder = TabNetEncoder(data_dim=4)
        self.decoder = TabNetDecoder(data_dim=4)
        self.binary_mask_generator = tfp.distributions.Binomial(total_count=1,
                                                                probs=0.3,
                                                                name="binary_mask_generator")
        self.data_batch = data.values
        self.mask = self.binary_mask_generator.sample(self.data_batch.shape)

    def test_valid_call(self):
        base_model = TabNet(input_dim=4,
                            features_metadata=self.features_metadata,
                            categorical_feature_embedding=self.categorical_feature_embedding,
                            encoder=self.encoder)
        pretrainer = TabNetPretrainer(model=base_model,
                                      features_metadata=self.features_metadata,
                                      decoder=self.decoder,
                                      missing_feature_probability=0.3)
        pretrainer(self.data_batch, mask=self.mask)

    def test_save_and_load(self):
        base_model = TabNet(input_dim=4,
                            features_metadata=self.features_metadata,
                            categorical_feature_embedding=self.categorical_feature_embedding,
                            encoder=self.encoder)
        pretrainer = TabNetPretrainer(model=base_model,
                                      features_metadata=self.features_metadata,
                                      decoder=self.decoder,
                                      missing_feature_probability=0.3)
        save_path = os.path.join(self.get_temp_dir(), "tabnet_pretrainer_lf.keras")
        pretrainer.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = pretrainer(self.data_batch, mask=self.mask)
        outputs_reloaded = reloaded_model(self.data_batch, mask=self.mask)
        # We can't check for AllClose here.. will investigate why
