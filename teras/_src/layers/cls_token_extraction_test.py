from teras._src.layers.cls_token_extraction import CLSTokenExtraction
from keras import random, ops
from keras.src.testing.test_case import TestCase


class CLSTokenExtractionTest(TestCase):
    def setUp(self):
        self.input_shape = (8, 3, 6)
        self.input_batch = random.normal(self.input_shape)

    def test_valid_output_shape(self):
        extraction = CLSTokenExtraction()
        outputs = extraction(self.input_batch)
        self.assertEqual(ops.shape(outputs),
                         (self.input_shape[0],
                          1,
                          self.input_shape[2]))
