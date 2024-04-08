from keras import random, ops
from teras._src.layers.tabnet.feature_transformer_layer import TabNetFeatureTransformerLayer
from keras.src.testing.test_case import TestCase


class TabNetFeatureTransformerLayerTest(TestCase):
    def setUp(self):
        self.batch_size = 16
        self.input_batch = random.normal((self.batch_size, 5))

    def test_valid_output_shape(self):
        layer = TabNetFeatureTransformerLayer(dim=8,
                                              batch_momentum=0.9)
        outputs = layer(self.input_batch)
        self.assertEqual(ops.shape(outputs), (self.batch_size, 8))

