from teras._src.layers.saint.reconstruction_head import SAINTReconstructionHead
from keras.src.testing.test_case import TestCase
from keras import ops, random
import numpy as np


class SAINTReconstructionHeadTest(TestCase):
    def setUp(self):
        self.input_batch = random.normal((8, 3, 16))
        # 0 represents the numerical features as required by the
        # `CategoricalEmbedding` layer
        self.cardinalities = [2, 10, 0]

    def test_valid_call(self):
        reconstruction_head = SAINTReconstructionHead(
            cardinalities=self.cardinalities,
            embedding_dim=16,
        )
        outputs = reconstruction_head(self.input_batch)

    def test_valid_output_shape(self):
        reconstruction_head = SAINTReconstructionHead(
            cardinalities=self.cardinalities,
            embedding_dim=16,
        )
        outputs = reconstruction_head(self.input_batch)
        self.assertEqual(ops.shape(outputs),
                         (8, sum(self.cardinalities) + 1))
