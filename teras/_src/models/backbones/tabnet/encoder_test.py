import os

import keras
from keras import random, ops
from keras.src.testing.test_case import TestCase

from teras._src.models.backbones.tabnet.encoder import TabNetEncoderBackbone


class TabNetEncoderBackboneTest(TestCase):
    def setUp(self):
        self.batch_size = 16
        self.input_batch = random.normal((self.batch_size, 5))

    # def test_valid_call(self):
    #     model = TabNetEncoderBackbone(input_dim=5,
    #                                   feature_transformer_dim=16,
    #                                   decision_step_dim=8)
    #     outputs = model(self.input_batch)
    #
    # def test_valid_output_shape(self):
    #     model = TabNetEncoderBackbone(input_dim=5,
    #                                   feature_transformer_dim=16,
    #                                   decision_step_dim=8)
    #     outputs = model(self.input_batch)
    #     self.assertEqual(ops.shape(outputs), (self.batch_size, 8))

    def test_model_save_and_load(self):
        model = TabNetEncoderBackbone(input_dim=5,
                                      feature_transformer_dim=16,
                                      decision_step_dim=8)
        outputs = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(),
                                 "tabnet_encoder_backbone.keras"
                                 )
        model.save(save_path)
        print("Reloading model...")
        reloaded_model = keras.models.load_model(save_path)

        # Check we got the real object back
        self.assertIsInstance(reloaded_model, TabNetEncoderBackbone)

        # Check that output matches
        reloaded_outputs = reloaded_model(self.input_batch)
        self.assertAllClose(
            ops.convert_to_numpy(outputs),
            ops.convert_to_numpy(reloaded_outputs)
        )

