from keras import ops, random
from teras._src.layers.categorical_embedding import CategoricalEmbedding
from keras.src.testing.test_case import TestCase


class CategoricalEmbeddingTest(TestCase):
    def setUp(self):
        self.input_batch = ops.array([ops.repeat(ops.array([1., 2.]), 5),
                                      ops.arange(0., 10.),
                                      random.normal((10,)),
                                      ])
        self.input_batch = ops.transpose(self.input_batch)
        # 0 represents the numerical features as required by the
        # `CategoricalEmbedding` layer
        self.cardinalities = [2, 10, 0]

    def test_valid_output_shape(self):
        ce = CategoricalEmbedding(embedding_dim=16,
                                  cardinalities=self.cardinalities)
        embeddings = ce(self.input_batch)
        expected_shape = (10, 2, 16) # (batch_size, num_categorical_features, embedding_dim)
        self.assertEqual(expected_shape, ops.shape(embeddings))

    def test_valid_output_shape_when_only_one_categorical(self):
        input_batch = ops.array([random.normal((10,)),
                                 random.normal((10,)),
                                 ops.repeat(ops.array([1., 2.]), 5),
                                 ])
        input_batch = ops.transpose(input_batch)
        cardinalities = [0, 0, 2]
        ce = CategoricalEmbedding(embedding_dim=16,
                                  cardinalities=cardinalities)
        embeddings = ce(input_batch)
        expected_shape = (10, 1, 16)
        self.assertEqual(expected_shape, ops.shape(embeddings))
