from teras.models import (TabTransformer,
                          TabTransformerClassifier,
                          TabTransformerRegressor,
                          TabTransformerPretrainer)
from teras.utils import get_features_metadata_for_embedding
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os


class TabTransformerClassifierTest(tf.test.TestCase):
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
        self.data_batch = data.values

    def test_valid_call(self):
        model = TabTransformerClassifier(features_metadata=self.features_metadata)
        model(self.data_batch)

    def test_save_and_load(self):
        model = TabTransformerClassifier(features_metadata=self.features_metadata)
        save_path = os.path.join(self.get_temp_dir(), "tabtransformer_classifier.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(self.data_batch)
        outputs_reloaded = reloaded_model(self.data_batch)
        self.assertAllClose(outputs_original, outputs_reloaded)


class TabTransformerRegressorTest(tf.test.TestCase):
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
        self.data_batch = data.values

    def test_valid_call(self):
        model = TabTransformerRegressor(features_metadata=self.features_metadata)
        model(self.data_batch)

    def test_save_and_load(self):
        model = TabTransformerRegressor(features_metadata=self.features_metadata)
        save_path = os.path.join(self.get_temp_dir(), "tabtransformer_regressor.keras")
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
        base_model = TabTransformer(features_metadata=self.features_metadata)
        pretrainer = TabTransformerPretrainer(model=base_model)
        pretrainer(self.data_batch, mask=self.mask)

    def test_save_and_load(self):
        base_model = TabTransformer(features_metadata=self.features_metadata)
        pretrainer = TabTransformerPretrainer(model=base_model)
        save_path = os.path.join(self.get_temp_dir(), "tabtransformer_pretrainer.keras")
        pretrainer.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = pretrainer(self.data_batch, mask=self.mask)
        outputs_reloaded = reloaded_model(self.data_batch, mask=self.mask)
        # We can't check for AllClose because the call method randomly shuffles the inputs
