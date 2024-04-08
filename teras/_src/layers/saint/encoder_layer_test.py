from teras._src.layers.saint.encoder_layer import SAINTEncoderLayer
from keras.src.testing.test_case import TestCase
from keras import ops, random


class SAINTEncoderLayerTest(TestCase):
    def setUp(self):
        self.input_shape = (8, 5, 16)
        self.input_batch = random.normal(self.input_shape)
        self.embedding_dim = self.input_shape[-1]

    def test_valid_output_shape(self):
        encoder = SAINTEncoderLayer(embedding_dim=self.embedding_dim)
        outputs = encoder(self.input_batch)
        self.assertEqual(ops.shape(outputs), self.input_shape)
    