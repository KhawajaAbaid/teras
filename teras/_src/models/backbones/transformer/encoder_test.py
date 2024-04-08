import os

import keras.models
from keras import random, ops
from keras.src.testing.test_case import TestCase

from teras._src.models.backbones.transformer.encoder import \
    TransformerEncoderBackbone


class TransformerEncoderTest(TestCase):
    def setUp(self):
        _batch_size, _input_dim, _batch_size = 8, 5, 16
        self.input_shape = (_batch_size, _input_dim, _batch_size)
        self.input_batch = random.normal(self.input_shape)

    def test_valid_call(self):
        model = TransformerEncoderBackbone(
            input_dim=self.input_shape[1],
            num_layers=6,
            embedding_dim=self.input_shape[-1],
            num_heads=8,
            feedforward_dim=64)
        outputs = model(self.input_batch)

    def test_valid_output_shape(self):
        model = TransformerEncoderBackbone(
            input_dim=self.input_shape[1],
            num_layers=6,
            embedding_dim=self.input_shape[-1],
            num_heads=8,
            feedforward_dim=64)
        outputs = model(self.input_batch)
        self.assertEqual(self.input_shape, ops.shape(outputs))

    def test_model_save_and_load(self):
        model = TransformerEncoderBackbone(
            input_dim=self.input_shape[1],
            num_layers=6,
            embedding_dim=self.input_shape[-1],
            num_heads=8,
            feedforward_dim=64)
        outputs = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(),
                                 "transformer_encoder_backbone.keras"
                                 )
        model.save(save_path)
        reloaded_model = keras.models.load_model(save_path)

        # Check we got the real object back
        self.assertIsInstance(reloaded_model, TransformerEncoderBackbone)

        # Check that output matches
        reloaded_outputs = reloaded_model(self.input_batch)
        self.assertAllClose(
            ops.convert_to_numpy(outputs),
            ops.convert_to_numpy(reloaded_outputs)
        )
