from teras.layers.cls_token_extraction import CLSTokenExtraction
from keras import random, ops
from keras.src.testing.test_case import TestCase


class CLSTokenExtractionTest(TestCase):
    def setUp(self):
        self.input_batch = random.normal((8, 3, 6))

    def test_valid_output_shape(self):
        extraction = CLSTokenExtraction()
        outputs = extraction(self.input_batch)
        self.assertEqual(ops.shape(outputs), (8, 1, 6))
