from keras import ops, random
from teras.layers.numerical_extractor import NumericalExtractor
from keras.src.testing.test_case import TestCase


class NumericalExtractorTest(TestCase):
    def setUp(self):
        self.input_batch = ops.array([random.normal((16,)),
                                      ops.repeat([1., 2., 3., 4.], 4),
                                      random.normal((16,)),
                                      ops.repeat([10., 20.], 8),
                                      random.normal((16,))])
        self.input_batch = ops.transpose(self.input_batch)
        self.numerical_idx = [0, 2, 4]

    def test_valid_output_shape(self):
        extractor = NumericalExtractor(numerical_idx=self.numerical_idx)
        numerical_features = extractor(self.input_batch)
        self.assertEqual((16, 3), ops.shape(numerical_features))
