from keras.src.testing.test_case import TestCase
from keras import ops, random
from teras._src.layers.transformer.feedforward import TransformerFeedForward


class TransformerFeedForwardTest(TestCase):
    def test_valid_output_shape(self):
        input_shape = (4, 5, 16)
        inputs = random.normal(shape=input_shape)
        encoder_layer = TransformerFeedForward(embedding_dim=16,
                                               hidden_dim=64
                                               )
        outputs = encoder_layer(inputs)
        self.assertEqual(ops.shape(outputs), input_shape)
