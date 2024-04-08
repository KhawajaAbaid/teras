import keras
from keras import random, ops
from keras.src.testing.test_case import TestCase

from teras._src.models.backbones.saint.saint import SAINTBackbone
from teras._src.models.pretrainers.saint import SAINTPretrainer


class SAINTPretrainerTest(TestCase):
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
        self.input_dim = 7
        self.embedding_dim = 16

    def test_valid_call(self):
        model = SAINTBackbone(
            input_dim=7,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim,
            embedd_inputs=False,
            return_cls_token_only=False,
        )
        pretrainer = SAINTPretrainer(
            model=model,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim
        )
        outputs = pretrainer(self.input_batch)

    def test_valid_output_shape(self):
        model = SAINTBackbone(
            input_dim=7,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim,
            embedd_inputs=False,
            return_cls_token_only=False,
        )
        pretrainer = SAINTPretrainer(
            model=model,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim
        )
        # Expected: outputs = (z_real, z_prime), reconstructed
        outputs = pretrainer(self.input_batch)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(len(outputs[0]), 2)

    def test_valid_encodings_shape(self):
        model = SAINTBackbone(
            input_dim=7,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim,
            embedd_inputs=False,
            return_cls_token_only=False,
        )
        pretrainer = SAINTPretrainer(
            model=model,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim,
        )
        outputs = pretrainer(self.input_batch)
        (z_real, z_mixed), reconstructed = outputs
        expected_z_shape = (
            self.batch_size,
            # +1 for the CLS feature
            (self.embedding_dim * self.input_dim // 2) * (self.input_dim
                                                          + 1)
        )
        self.assertEqual(ops.shape(z_real), expected_z_shape)
        self.assertEqual(ops.shape(z_mixed), expected_z_shape)

    def test_valid_reconstructed_features_shape(self):
        model = SAINTBackbone(
            input_dim=7,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim,
            embedd_inputs=False,
            return_cls_token_only=False,
        )
        pretrainer = SAINTPretrainer(
            model=model,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim
        )
        outputs = pretrainer(self.input_batch)
        (z_real, z_mixed), reconstructed = outputs
        expected_reconstructed_shape = (
            self.batch_size,
            sum(self.cardinalities) + self.num_continuous)
        self.assertEqual(ops.shape(reconstructed),
                         expected_reconstructed_shape)

    def test_fit(self):
        model = SAINTBackbone(
            input_dim=7,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim,
            embedd_inputs=False,
            return_cls_token_only=False,
        )
        pretrainer = SAINTPretrainer(
            model=model,
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim
        )
        pretrainer.compile(optimizer=keras.optimizers.Adam())
        pretrainer.build(ops.shape(self.input_batch))
        pretrainer.fit(self.input_batch)
