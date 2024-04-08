from keras import ops, random
from teras._src.layers.continuous_extraction import ContinuousExtraction
from keras.src.testing.test_case import TestCase


class ContinuousExtractionTest(TestCase):
    def setUp(self):
        self.input_batch = ops.array(
            [random.normal((16,)),
             ops.repeat(ops.array([1., 2., 3., 4.]), 4),
             random.normal((16,)),
             ops.repeat(ops.array([10., 20.]), 8),
             random.normal((16,))])
        self.input_batch = ops.transpose(self.input_batch)
        self.continuous_idx = [0, 2, 4]

    def test_valid_output_shape(self):
        extraction = ContinuousExtraction(
            continuous_idx=self.continuous_idx)
        continuous_features = extraction(self.input_batch)
        self.assertEqual((16, 3), ops.shape(continuous_features))
