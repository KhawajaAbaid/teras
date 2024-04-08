from keras import random, ops
from teras._src.layers.tabnet.feature_transformer import TabNetFeatureTransformer
from keras.src.testing.test_case import TestCase


class TabNetFeatureTransformerTest(TestCase):
    def setUp(self):
        self.batch_size = 16
        self.input_batch = random.normal((self.batch_size, 5))

    def test_valid_output_shape(self):
        layer = TabNetFeatureTransformer(hidden_dim=8,
                                         num_shared_layers=2,
                                         num_decision_dependent_layers=2,
                                         batch_momentum=0.99)
        outputs = layer(self.input_batch)
        self.assertEqual(ops.shape(outputs), (self.batch_size, 8))

