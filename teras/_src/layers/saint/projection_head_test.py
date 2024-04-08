from teras._src.layers.saint.projection_head import SAINTProjectionHead
from keras.src.testing.test_case import TestCase
from keras import ops, random


class SAINTProjectionHeadTest(TestCase):
    def setUp(self):
        self.input_batch = random.normal((8, 5, 16))
        self.embedding_dim = 16

    def test_valid_call(self):
        projection_head = SAINTProjectionHead(
            hidden_dim=16,
            output_dim=10,
        )
        outputs = projection_head(self.input_batch)

    def test_valid_output_shape(self):
        projection_head = SAINTProjectionHead(
            hidden_dim=16,
            output_dim=10,
        )
        outputs = projection_head(self.input_batch)
        self.assertEqual(ops.shape(outputs), (8, 5, 10))
