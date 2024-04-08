import os

import keras
from keras import random, ops
from keras.src.testing.test_case import TestCase

from teras._src.models.autoencoders.tvae.decoder import TVAEDecoder


class TVAEDecoderTest(TestCase):
    def setUp(self):
        self.batch_size = 8
        self.data_dim = 5
        self.inputs = random.normal((self.batch_size, 8))

    def test_valid_call(self):
        decoder = TVAEDecoder(data_dim=self.data_dim)
        outputs = decoder(self.inputs)

    def test_valid_output_shape(self):
        decoder = TVAEDecoder(data_dim=self.data_dim)
        outputs = decoder(self.inputs)
        # Decoder returns generated samples and sigmas
        self.assertEqual(len(outputs), 2)
        # Generated samples shape check
        self.assertEqual(ops.shape(outputs[0]),
                         (self.batch_size, self.data_dim))
        # sigmas shape check
        self.assertEqual(ops.shape(outputs[1]),
                         (self.data_dim,))

    def test_model_save_and_load(self):
        decoder = TVAEDecoder(data_dim=self.data_dim)
        outputs = decoder(self.inputs)
        save_path = os.path.join(self.get_temp_dir(),
                                 "tvae_decoder.keras"
                                 )
        decoder.save(save_path)
        reloaded_model = keras.models.load_model(save_path)

        # Check we got the real object back
        self.assertIsInstance(reloaded_model, TVAEDecoder)

        # Check that output matches
        reloaded_outputs = reloaded_model(self.inputs)
        for original, reloaded in zip(outputs, reloaded_outputs):
            self.assertAllClose(
                ops.convert_to_numpy(original),
                ops.convert_to_numpy(reloaded)
            )
