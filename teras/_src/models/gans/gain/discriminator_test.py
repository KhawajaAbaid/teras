import os

import keras
from keras import ops
from keras.src.testing.test_case import TestCase

from teras._src.models.gans.gain.discriminator import GAINDiscriminator


class GAINDiscriminatorTest(TestCase):
    def setUp(self):
        self.latent_inputs = ops.ones((8, 16))

    def test_valid_call(self):
        discriminator = GAINDiscriminator(data_dim=7)
        outputs = discriminator(self.latent_inputs)

    def test_valid_output_shape(self):
        discriminator = GAINDiscriminator(data_dim=7)
        outputs = discriminator(self.latent_inputs)
        assert ops.shape(outputs) == (8, 7)

    def test_save_and_load(self):
        discriminator = GAINDiscriminator(data_dim=7)
        outputs_original = discriminator(self.latent_inputs)
        save_path = os.path.join(self.get_temp_dir(),
                                 "gain_discriminator.keras")
        discriminator.save(save_path)
        reloaded_model = keras.models.load_model(save_path)
        outputs_reloaded = reloaded_model(self.latent_inputs)
        self.assertAllClose(ops.convert_to_numpy(outputs_original),
                            ops.convert_to_numpy(outputs_reloaded))
