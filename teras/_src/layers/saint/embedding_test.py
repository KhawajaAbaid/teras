from teras._src.layers.saint.embedding import SAINTEmbedding
from keras.src.testing.test_case import TestCase
from keras import ops, random


class SAINTEmbeddingTest(TestCase):
    def setUp(self):
        # we assume there are 4 features in the dataset, with feature 1
        # having 5 unique categories, feature 2 having 3 unique
        # categories, feature 3 with 2 unique categories and feature 4
        # with 7 unique categories
        x_cat = ops.array([
            ops.cast(random.randint((16,), minval=0, maxval=card),
                     dtype="float32")
            for card in [5, 3, 2, 7]]
        )
        x_cat = ops.transpose(x_cat)
        x_cont = random.normal((16, 3), dtype="float32")
        self.input_batch = ops.concatenate([x_cat, x_cont], axis=1)
        self.cardinalities = [5, 3, 2, 7, 0, 0, 0]  # 0 for continuous

    def test_valid_output_shape(self):
        embedding = SAINTEmbedding(embedding_dim=8,
                                   cardinalities=self.cardinalities)
        outputs = embedding(self.input_batch)
        self.assertEqual(ops.shape(outputs), (16, 7, 8))
