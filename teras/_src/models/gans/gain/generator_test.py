import os

import keras
from keras import random, ops
from keras.src.testing.test_case import TestCase

from teras._src.models.gans.gain.generator import GAINGenerator


class GAINGeneratorTest(TestCase):
    def setUp(self):
        self.latent_inputs = random.normal((8, 16))

    def test_valid_call(self):
        generator = GAINGenerator(data_dim=7)
        outputs = generator(self.latent_inputs)

    def test_valid_output_shape(self):
        generator = GAINGenerator(data_dim=7)
        outputs = generator(self.latent_inputs)
        assert ops.shape(outputs) == (8, 7)

    def test_save_and_load(self):
        generator = GAINGenerator(data_dim=7)
        outputs_original = generator(self.latent_inputs)
        save_path = os.path.join(self.get_temp_dir(), "gain_generator.keras")
        generator.save(save_path)
        reloaded_model = keras.models.load_model(save_path)
        outputs_reloaded = reloaded_model(self.latent_inputs)
        self.assertAllClose(ops.convert_to_numpy(outputs_original),
                            ops.convert_to_numpy(outputs_reloaded))

