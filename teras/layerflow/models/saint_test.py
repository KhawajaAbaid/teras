import tensorflow as tf
from tensorflow import keras
from teras.layerflow.models.saint import (SAINT,
                                          SAINTPretrainer)
from teras.layers import (CategoricalFeatureEmbedding,
                          SAINTNumericalFeatureEmbedding,
                          SAINTEncoder,
                          MixUp,
                          CutMix,
                          SAINTReconstructionHead,
                          SAINTProjectionHead)
from teras.utils import get_features_metadata_for_embedding
import pandas as pd
import numpy as np
import os


class SAINTTest(tf.test.TestCase):
    def setUp(self):
        data = pd.DataFrame({"income": np.ones(10),
                             "goals": np.ones(10),
                             'player_level': [5, 7, 9, 8, 9, 10, 9, 7, 8, 9],
                             'shirt_number': [7, 10, 10, 7, 7, 10, 10, 10, 7, 10]})
        categorical_feats = ["player_level", "shirt_number"]
        numerical_feats = ["income", "goals"]
        self.input_dim = 4
        self.features_metadata = get_features_metadata_for_embedding(pd.DataFrame(data),
                                                                     categorical_features=categorical_feats,
                                                                     numerical_features=numerical_feats)
        self.categorical_feature_embedding = CategoricalFeatureEmbedding(features_metadata=self.features_metadata,
                                                                         embedding_dim=32)
        self.numerical_feature_embedding = SAINTNumericalFeatureEmbedding(features_metadata=self.features_metadata,
                                                                          embedding_dim=32)
        self.encoder = SAINTEncoder(data_dim=self.input_dim)
        self.head = keras.layers.Dense(1)
        self.data_batch = data.values

    def test_valid_call(self):
        model = SAINT(input_dim=self.input_dim,
                      categorical_feature_embedding=self.categorical_feature_embedding,
                      numerical_feature_embedding=self.numerical_feature_embedding,
                      encoder=self.encoder,
                      head=self.head)
        model(self.data_batch)

    def test_save_and_load(self):
        model = SAINT(input_dim=self.input_dim,
                      categorical_feature_embedding=self.categorical_feature_embedding,
                      numerical_feature_embedding=self.numerical_feature_embedding,
                      encoder=self.encoder,
                      head=self.head)
        save_path = os.path.join(self.get_temp_dir(), "saint_lf.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(self.data_batch)
        outputs_reloaded = reloaded_model(self.data_batch)
        self.assertAllClose(outputs_original, outputs_reloaded)


class SAINTPretrainerTest(tf.test.TestCase):
    def setUp(self):
        data = pd.DataFrame({"income": np.ones(10),
                             "goals": np.ones(10),
                             'player_level': [5, 7, 9, 8, 9, 10, 9, 7, 8, 9],
                             'shirt_number': [7, 10, 10, 7, 7, 10, 10, 10, 7, 10]})
        categorical_feats = ["player_level", "shirt_number"]
        numerical_feats = ["income", "goals"]
        self.input_dim = 4
        self.features_metadata = get_features_metadata_for_embedding(pd.DataFrame(data),
                                                                     categorical_features=categorical_feats,
                                                                     numerical_features=numerical_feats)
        self.categorical_feature_embedding = CategoricalFeatureEmbedding(features_metadata=self.features_metadata,
                                                                         embedding_dim=32)
        self.numerical_feature_embedding = SAINTNumericalFeatureEmbedding(features_metadata=self.features_metadata,
                                                                          embedding_dim=32)
        self.encoder = SAINTEncoder(data_dim=self.input_dim)

        # layers for pretrainer
        self.mixup = MixUp()
        self.cutmix = CutMix()
        self.projection_head_1 = SAINTProjectionHead()
        self.projection_head_2 = SAINTProjectionHead()
        self.reconstruction_head = SAINTReconstructionHead(features_metadata=self.features_metadata)
        self.data_batch = data.values

    def test_valid_call(self):
        base_model = SAINT(input_dim=self.input_dim,
                           categorical_feature_embedding=self.categorical_feature_embedding,
                           numerical_feature_embedding=self.numerical_feature_embedding,
                           encoder=self.encoder)
        pretrainer = SAINTPretrainer(model=base_model,
                                     features_metadata=self.features_metadata,
                                     mixup=self.mixup,
                                     cutmix=self.cutmix,
                                     projection_head_1=self.projection_head_1,
                                     projection_head_2=self.projection_head_2,
                                     reconstruction_head=self.reconstruction_head)
        pretrainer(self.data_batch)

    def test_save_and_load(self):
        base_model = SAINT(input_dim=self.input_dim,
                           categorical_feature_embedding=self.categorical_feature_embedding,
                           numerical_feature_embedding=self.numerical_feature_embedding,
                           encoder=self.encoder)
        pretrainer = SAINTPretrainer(model=base_model,
                                     features_metadata=self.features_metadata,
                                     mixup=self.mixup,
                                     cutmix=self.cutmix,
                                     projection_head_1=self.projection_head_1,
                                     projection_head_2=self.projection_head_2,
                                     reconstruction_head=self.reconstruction_head)
        save_path = os.path.join(self.get_temp_dir(), "saint_pretrainer_lf.keras")
        pretrainer.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = pretrainer(self.data_batch)
        outputs_reloaded = reloaded_model(self.data_batch)
        # We can't check for AllClose
