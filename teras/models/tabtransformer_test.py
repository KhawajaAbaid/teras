from teras.models import (TabTransformerClassifier,
                          TabTransformerRegressor)
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
        model = TabTransformerClassifier(num_classes=2,
                                         input_dim=4,
                                         features_metadata=self.features_metadata)
        model(self.data_batch)

    def test_save_and_load(self):
        model = TabTransformerClassifier(num_classes=2,
                                         input_dim=4,
                                         features_metadata=self.features_metadata)
        save_path = os.path.join(self.get_temp_dir(), "tabtransformer_classifier.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(self.data_batch)
        outputs_reloaded = reloaded_model(self.data_batch)
        self.assertAllClose(outputs_original, outputs_reloaded)
        self.assertAllClose(model.weights, reloaded_model.weights)


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
        model = TabTransformerRegressor(num_outputs=1,
                                        input_dim=4,
                                        features_metadata=self.features_metadata)
        model(self.data_batch)

    def test_save_and_load(self):
        model = TabTransformerRegressor(num_outputs=1,
                                        input_dim=4,
                                        features_metadata=self.features_metadata)
        save_path = os.path.join(self.get_temp_dir(), "tabtransformer_regressor.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(self.data_batch)
        outputs_reloaded = reloaded_model(self.data_batch)
        self.assertAllClose(outputs_original, outputs_reloaded)
