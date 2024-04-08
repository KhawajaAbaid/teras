import keras
from keras import random, ops
from keras.src.testing.test_case import TestCase

from teras._src.models.backbones.tab_transformer.tab_transformer import \
    TabTransformerBackbone
from teras._src.models.pretrainers.tab_transformer import (
    TabTransformerMLMPretrainer,
    TabTransformerRTDPretrainer
)


class TabTransformerMLMPretrainerTest(TestCase):
    def setUp(self):
        # we assume there are 4 categorical features in the dataset,
        # with feature 1 having 5 unique categories, feature 2 having 3
        # unique categories, feature 3 with 2 unique categories and
        # feature 4 with 7 unique categories
        self.batch_size = 16
        x_cat = ops.array([
            random.randint((self.batch_size,), minval=0, maxval=card)
            for card in [5, 3, 2, 7]],
            dtype="float32")
        x_cat = ops.transpose(x_cat)
        x_cont = random.normal((self.batch_size, 3), dtype="float32")
        self.input_batch = ops.concatenate([x_cat, x_cont], axis=1)
        self.cardinalities = [5, 3, 2, 7, 0, 0, 0]  # 0 for continuous
        self.num_categorical = 4    # Number of categorical features
        self.num_continuous = 3     # Number of continuous features
        self.embedding_dim = 16
        self.mask = random.binomial((self.batch_size, 7),
                                    1, 0.5)

    def test_valid_call(self):
        tab_model = TabTransformerBackbone(
            input_dim=7,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim
        )
        pretrainer = TabTransformerMLMPretrainer(
            model=tab_model,
            data_dim=7
        )
        reconstructed_features = pretrainer(self.input_batch,
                                            self.mask)

    def test_valid_output_shape(self):
        tab_model = TabTransformerBackbone(
            input_dim=7,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim
        )
        pretrainer = TabTransformerMLMPretrainer(
            model=tab_model,
            data_dim=7
        )
        reconstructed_features = pretrainer(self.input_batch,
                                            self.mask)
        self.assertEqual(ops.shape(reconstructed_features),
                         (self.batch_size, 7))

    def test_fit(self):
        tab_model = TabTransformerBackbone(
            input_dim=7,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim
        )
        pretrainer = TabTransformerMLMPretrainer(
            model=tab_model,
            data_dim=7
        )
        # pretrainer.compile(loss=keras.losses.CategoricalCrossentropy(
        #     from_logits=True),
        #     optimizer=(keras.optimizers.Adam()))
        pretrainer.compile()
        pretrainer.build(ops.shape(self.input_batch))
        pretrainer.fit(self.input_batch)


class TabTransformerRTDPretrainerTest(TestCase):
    def setUp(self):
        # we assume there are 4 categorical features in the dataset,
        # with feature 1 having 5 unique categories, feature 2 having 3
        # unique categories, feature 3 with 2 unique categories and
        # feature 4 with 7 unique categories
        self.batch_size = 16
        x_cat = ops.array([
            random.randint((self.batch_size,), minval=0, maxval=card)
            for card in [5, 3, 2, 7]],
            dtype="float32")
        x_cat = ops.transpose(x_cat)
        x_cont = random.normal((self.batch_size, 3), dtype="float32")
        self.input_batch = ops.concatenate([x_cat, x_cont], axis=1)
        self.cardinalities = [5, 3, 2, 7, 0, 0, 0]  # 0 for continuous
        self.num_categorical = 4    # Number of categorical features
        self.num_continuous = 3     # Number of continuous features
        self.embedding_dim = 16
        self.mask = random.binomial((self.batch_size, 7),
                                    1, 0.5)

    def test_valid_call(self):
        tab_model = TabTransformerBackbone(
            input_dim=7,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim
        )
        pretrainer = TabTransformerRTDPretrainer(
            model=tab_model,
            data_dim=7
        )
        predicted_mask = pretrainer(self.input_batch,
                                    self.mask)

    def test_valid_output_shape(self):
        tab_model = TabTransformerBackbone(
            input_dim=7,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim
        )
        pretrainer = TabTransformerRTDPretrainer(
            model=tab_model,
            data_dim=7
        )
        predicted_mask = pretrainer(self.input_batch,
                                    self.mask)
        self.assertEqual(ops.shape(predicted_mask),
                         (self.batch_size, 7))

    def test_fit(self):
        tab_model = TabTransformerBackbone(
            input_dim=7,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim
        )
        pretrainer = TabTransformerRTDPretrainer(
            model=tab_model,
            data_dim=7
        )
        pretrainer.compile(optimizer=keras.optimizers.Adam())
        pretrainer.build(ops.shape(self.input_batch))
        pretrainer.fit(self.input_batch)
