from keras import random, ops
from teras._src.layers.tabnet.attentive_transformer import TabNetAttentiveTransformer
from keras.src.testing.test_case import TestCase


class TabNetAttentiveTransformerTest(TestCase):
    def setUp(self):
        self.batch_size = 16
        self.data_dim = 5
        self.hidden_representations = random.normal((self.batch_size, 8))

    def test_valid_output_shape(self):
        layer = TabNetAttentiveTransformer(data_dim=self.data_dim,
                                           batch_momentum=0.99)
        prior_scales = random.normal((self.batch_size, self.data_dim))
        outputs = layer(self.hidden_representations,
                        prior_scales=prior_scales)
        self.assertEqual(ops.shape(outputs),
                         (self.batch_size, self.data_dim))

    def test_valid_output_shape_with_prior_scales(self):
        layer = TabNetAttentiveTransformer(data_dim=self.data_dim,
                                           batch_momentum=0.99)
        prior_scales = random.normal((self.batch_size, self.data_dim))
        outputs = layer(self.hidden_representations,
                        prior_scales=prior_scales)
        self.assertEqual(ops.shape(outputs),
                         (self.batch_size, self.data_dim))
