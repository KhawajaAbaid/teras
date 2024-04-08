import os

import keras
from keras import random, ops
from keras.src.testing.test_case import TestCase
from teras._src.models.pretrainers.tabnet.decoder import TabNetDecoder


class TabNetDecoderTest(TestCase):
    def setUp(self):
        self.batch_size = 16
        self.encoded_features = random.normal((self.batch_size, 8))

    def test_valid_call(self):
        model = TabNetDecoder(data_dim=5,
                              feature_transformer_dim=16,
                              decision_step_dim=8)
        reconstructed_features = model(self.encoded_features)

    def test_valid_output_shape(self):
        model = TabNetDecoder(data_dim=5,
                              feature_transformer_dim=16,
                              decision_step_dim=8)
        reconstructed_features = model(self.encoded_features)
        self.assertEqual(ops.shape(reconstructed_features),
                         (self.batch_size, 5))

    def test_model_save_and_load(self):
        model = TabNetDecoder(data_dim=5,
                              feature_transformer_dim=16,
                              decision_step_dim=8)
        reconstructed_features = model(self.encoded_features)
        save_path = os.path.join(self.get_temp_dir(),
                                 "tabnet_decoder_backbone.keras"
                                 )
        model.save(save_path)
        reloaded_model = keras.models.load_model(save_path)

        # Check we got the real object back
        self.assertIsInstance(reloaded_model, TabNetDecoder)

        # Check that output matches
        reloaded_outputs = reloaded_model(self.encoded_features)
        self.assertAllClose(
            ops.convert_to_numpy(reconstructed_features),
            ops.convert_to_numpy(reloaded_outputs)
        )
