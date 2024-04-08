from keras import ops, random
from teras._src.layers.categorical_extraction import CategoricalExtraction
from keras.src.testing.test_case import TestCase
from teras._src.utils import compute_cardinalities


class CategoricalExtractionTest(TestCase):
    def setUp(self):
        self.input_batch = ops.array(
            [random.normal((16,)),
             ops.repeat(ops.array([1., 2., 3., 4.]), 4),
             random.normal((16,)),
             ops.repeat(ops.array([10., 20.]), 8),
             random.normal((16,))])
        self.input_batch = ops.transpose(self.input_batch)
        self.categorical_idx = [1, 3]
        self.categorical_cardinalities = [4, 2]

    def test_valid_output_shape(self):
        extraction = CategoricalExtraction(
            categorical_idx=self.categorical_idx)
        categorical_features = extraction(self.input_batch)
        self.assertEqual((16, 2), ops.shape(categorical_features))

    def test_valid_output(self):
        extraction = CategoricalExtraction(
            categorical_idx=self.categorical_idx)
        categorical_features = extraction(self.input_batch)
        categorical_features = ops.convert_to_numpy(categorical_features)
        cards = compute_cardinalities(
            categorical_features,
            categorical_idx=list(range(categorical_features.shape[1])))
        cards = list(cards)
        self.assertEqual(self.categorical_cardinalities, cards)
