from keras.src.testing.test_case import TestCase
from keras import ops, random
from teras.layers.ctgan.discriminator_layer import CTGANDiscriminatorLayer


class CTGANDiscriminatorLayerTest(TestCase):
    def setUp(self):
        self.input_shape = (4, 8)
        self.inputs = random.normal(shape=self.input_shape)

    def test_valid_call(self):
        encoder_layer = CTGANDiscriminatorLayer(dim=16)
        outputs = encoder_layer(self.inputs)

    def test_valid_output_shape(self):
        encoder_layer = CTGANDiscriminatorLayer(dim=16)
        outputs = encoder_layer(self.inputs)
        self.assertEqual(ops.shape(outputs), (4, 16))
