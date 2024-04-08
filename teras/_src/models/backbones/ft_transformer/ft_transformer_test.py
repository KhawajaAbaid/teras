import os

import keras
from keras import random, ops
from keras.src.testing.test_case import TestCase

from teras._src.models.backbones.ft_transformer.ft_transformer import \
    FTTransformerBackbone


class FTTransformerBackboneTest(TestCase):
    def setUp(self):
        # we assume there are 4 categorical features in the dataset,
        # with feature 1 having 5 unique categories, feature 2 having 3
        # unique categories, feature 3 with 2 unique categories and
        # feature 4 with 7 unique categories
        x_cat = ops.array([
            random.randint((16,), minval=0, maxval=card)
            for card in [5, 3, 2, 7]],
            dtype="float32")
        x_cat = ops.transpose(x_cat)
        x_cont = random.normal((16, 3), dtype="float32")
        self.input_batch = ops.concatenate([x_cat, x_cont], axis=1)
        self.cardinalities = [5, 3, 2, 7, 0, 0, 0]  # 0 for continuous
        self.num_categorical = 4 # Number of categorical features
        self.num_continuous = 3 # Number of continuous features
        self.embedding_dim = 16

    def test_valid_call(self):
        model = FTTransformerBackbone(
            input_dim=self.input_batch.shape[1],
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim
        )
        outputs = model(self.input_batch)

    def test_valid_output_shape(self):
        model = FTTransformerBackbone(
            input_dim=self.input_batch.shape[1],
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim
        )
        outputs = model(self.input_batch)
        # model returns embeddings for the cls token only!
        true_output_shape = (self.input_batch.shape[0],
                             1,
                             self.embedding_dim)
        self.assertEqual(ops.shape(outputs), true_output_shape)

    def test_model_save_and_load(self):
        model = FTTransformerBackbone(
            input_dim=self.input_batch.shape[1],
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim
        )
        outputs = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(),
                                 "ft_transformer_backbone.keras"
                                 )
        model.save(save_path)
        reloaded_model = keras.models.load_model(save_path)

        # Check we got the real object back
        self.assertIsInstance(reloaded_model, FTTransformerBackbone)

        # Check that output matches
        reloaded_outputs = reloaded_model(self.input_batch)
        self.assertAllClose(
            ops.convert_to_numpy(outputs),
            ops.convert_to_numpy(reloaded_outputs)
        )
