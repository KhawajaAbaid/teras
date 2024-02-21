from keras import random, ops
from teras.models.backbones.tabnet.encoder import TabNetEncoderBackbone
from keras.src.testing.test_case import TestCase


class TabNetEncoderBackboneTest(TestCase):
    def setUp(self):
        self.batch_size = 16
        self.input_batch = random.normal((self.batch_size, 5))

    def test_valid_call(self):
        model = TabNetEncoderBackbone(data_dim=5,
                                      feature_transformer_dim=8,
                                      decision_step_dim=8)
        outputs = model(self.input_batch)

    def test_valid_output_shape(self):
        model = TabNetEncoderBackbone(data_dim=5,
                                      feature_transformer_dim=8,
                                      decision_step_dim=8)
        outputs = model(self.input_batch)
        self.assertEqual(ops.shape(outputs), (self.batch_size, 8))
