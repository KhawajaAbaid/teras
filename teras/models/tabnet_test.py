from teras.models.tabnet import TabNetClassifier, TabNetRegressor, TabNetPretrainer, TabNet
from teras.utils import get_features_metadata_for_embedding
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
import tensorflow_probability as tfp


class TabNetClassifierTest(tf.test.TestCase):
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
        model = TabNetClassifier(num_classes=2,
                                 input_dim=4,
                                 features_metadata=self.features_metadata)
        model(self.data_batch)

    def test_save_and_load(self):
        model = TabNetClassifier(num_classes=2,
                                 input_dim=4,
                                 features_metadata=self.features_metadata)
        save_path = os.path.join(self.get_temp_dir(), "tabnet_classifier.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(self.data_batch)
        outputs_reloaded = reloaded_model(self.data_batch)
        self.assertAllClose(outputs_original, outputs_reloaded)
        self.assertAllClose(model.weights, reloaded_model.weights)


class TabNetRegressorTest(tf.test.TestCase):
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
        model = TabNetRegressor(num_outputs=1,
                                input_dim=4,
                                features_metadata=self.features_metadata)
        model(self.data_batch)

    def test_save_and_load(self):
        model = TabNetRegressor(num_outputs=1,
                                input_dim=4,
                                features_metadata=self.features_metadata)
        save_path = os.path.join(self.get_temp_dir(), "tabnet_regressor.keras")
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
        self.input_dim = 4
        self.binary_mask_generator = tfp.distributions.Binomial(total_count=1,
                                                                probs=0.3,
                                                                name="binary_mask_generator")
        self.data_batch = data.values
        self.mask = self.binary_mask_generator.sample(self.data_batch.shape)

    def test_valid_call(self):
        base_model = TabNet(input_dim=self.input_dim,
                            features_metadata=self.features_metadata)
        pretrainer = TabNetPretrainer(model=base_model,
                                      input_dim=self.input_dim,
                                      features_metadata=self.features_metadata,)
        pretrainer(self.data_batch, mask=self.mask)

    def test_save_and_load(self):
        base_model = TabNet(input_dim=self.input_dim,
                            features_metadata=self.features_metadata)
        pretrainer = TabNetPretrainer(model=base_model,
                                      input_dim=self.input_dim,
                                      features_metadata=self.features_metadata,)
        save_path = os.path.join(self.get_temp_dir(), "tabnet_pretrainer.keras")
        pretrainer.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = pretrainer(self.data_batch, mask=self.mask)
        outputs_reloaded = reloaded_model(self.data_batch, mask=self.mask)
