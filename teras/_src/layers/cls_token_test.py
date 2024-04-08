from teras._src.layers.cls_token import CLSToken
from keras.src.testing.test_case import TestCase
from keras import ops, random


class CLSTokenTest(TestCase):
    def setUp(self):
        self.input_shape = (16, 5, 8)
        self.input_batch = random.normal(self.input_shape)

    def test_valid_output_shape(self):
        cls_token = CLSToken(embedding_dim=8)
        outputs = cls_token(self.input_batch)
        self.assertEqual(
            ops.shape(outputs),
            (self.input_shape[0],
             self.input_shape[1] + 1,
             self.input_shape[-1]))
