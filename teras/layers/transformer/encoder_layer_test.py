from keras.src.testing.test_case import TestCase
from keras import ops, random
from teras.layers.transformer.encoder_layer import TransformerEncoderLayer


class TransformerEncoderLayerTest(TestCase):
    def test_valid_output_shape(self):
        input_shape = (4, 5, 16)
        inputs = random.normal(shape=input_shape)
        encoder_layer = TransformerEncoderLayer(embedding_dim=16,
                                                num_heads=8,
                                                feedforward_dim=64,
                                                )
        outputs = encoder_layer(inputs)
        self.assertEqual(ops.shape(outputs), input_shape)
