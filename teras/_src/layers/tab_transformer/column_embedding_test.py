from teras._src.layers.tab_transformer.column_embedding import TabTransformerColumnEmbedding
from keras import ops, random
from keras.src.testing.test_case import TestCase


class TabTransformerColumnEmbeddingTest(TestCase):
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
        column_embedding = TabTransformerColumnEmbedding(
            cardinalities=self.cardinalities,
            embedding_dim=32,
        )
        embeddings = column_embedding(self.input_batch)
        self.assertEqual((16, 4, 32), ops.shape(embeddings))

    def test_with_custom_shared_embedding_dim(self):
        column_embedding = TabTransformerColumnEmbedding(
            cardinalities=self.cardinalities,
            embedding_dim=32,
            shared_embedding_dim=8,
        )
        embeddings = column_embedding(self.input_batch)
        self.assertEqual((16, 4, 32), ops.shape(embeddings))

    def test_when_use_shared_embedding_is_false(self):
        column_embedding = TabTransformerColumnEmbedding(
            cardinalities=self.cardinalities,
            embedding_dim=32,
            use_shared_embedding=False,
        )
        embeddings = column_embedding(self.input_batch)
        self.assertEqual((16, 4, 32), ops.shape(embeddings))

    def test_when_join_method_is_add(self):
        column_embedding = TabTransformerColumnEmbedding(
            cardinalities=self.cardinalities,
            embedding_dim=32,
            join_method="add"
        )
        embeddings = column_embedding(self.input_batch)
        self.assertEqual((16, 4, 32), ops.shape(embeddings))
