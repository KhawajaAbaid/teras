import os

import keras
from keras import ops
from keras.src.testing.test_case import TestCase

from teras._src.models.gans.ctgan.discriminator import CTGANDiscriminator


class CTGANDiscriminatorTest(TestCase):
    def setUp(self):
        self.inputs = ops.ones((16, 5))
        self.packing_degree = 4

    def test_valid_call(self):
        discriminator = CTGANDiscriminator(packing_degree=self.packing_degree)
        outputs = discriminator(self.inputs)

    def test_valid_output_shape(self):
        discriminator = CTGANDiscriminator(packing_degree=self.packing_degree)
        outputs = discriminator(self.inputs)
        self.assertEqual(ops.shape(outputs), (16 // self.packing_degree, 1))

    def test_save_and_load(self):
        discriminator = CTGANDiscriminator(packing_degree=self.packing_degree)
        outputs_original = discriminator(self.inputs)
        save_path = os.path.join(self.get_temp_dir(),
                                 "ctgan_discriminator.keras")
        discriminator.save(save_path)
        reloaded_discriminator = keras.models.load_model(save_path)
        self.assertIsInstance(reloaded_discriminator, CTGANDiscriminator)
        outputs_reloaded = reloaded_discriminator(self.inputs)
        self.assertAllClose(ops.convert_to_numpy(outputs_original),
                            ops.convert_to_numpy(outputs_reloaded))
