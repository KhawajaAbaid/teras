import os

import keras
import pytest
from keras import backend
from keras import random, ops
from keras.src.testing.test_case import TestCase
from teras._src.preprocessing.data_samplers.ctgan import CTGANDataSampler
from teras._src.preprocessing.data_transformers.ctgan import CTGANDataTransformer
from teras._src.testing.markers import skip_on_torch

from teras._src.models.gans.ctgan.generator import CTGANGenerator
from teras._src.utils import generate_fake_gemstone_data

if backend.backend() == "torch":
    pytest.skip("Tests enter inifite loop on torch backend.",
                allow_module_level=True)


class CTGANGeneratorTest(TestCase):
    def setUp(self):
        fake_gem_df = generate_fake_gemstone_data(num_samples=32)
        num_cols = ["depth", "table"]
        cat_cols = ["cut", "color", "clarity"]
        data_transformer = CTGANDataTransformer(continuous_features=num_cols,
                                                categorical_features=cat_cols)
        x_transformed = data_transformer.fit_transform(fake_gem_df)

        data_sampler = CTGANDataSampler(batch_size=8,
                                        categorical_features=cat_cols,
                                        metadata=data_transformer.get_metadata())
        self.dataset = data_sampler.get_dataset(x_transformed=x_transformed,
                                                x_original=fake_gem_df)
        self.metadata = data_transformer.get_metadata()
        self.data_dim = data_sampler.data_dim

    def test_valid_call(self):
        z = random.uniform((8, 16))
        generator = CTGANGenerator(data_dim=self.data_dim,
                                   metadata=self.metadata)
        outputs = generator(z)

    def test_valid_output_shape(self):
        z = random.uniform((8, 16))
        generator = CTGANGenerator(data_dim=self.data_dim,
                                   metadata=self.metadata)
        outputs = generator(z)
        self.assertEqual(ops.shape(outputs), (8, self.data_dim))

    @skip_on_torch
    def test_save_and_load(self):
        z = random.uniform((8, 16))
        generator = CTGANGenerator(data_dim=self.data_dim,
                                   metadata=self.metadata)
        outputs_original = generator(z)
        save_path = os.path.join(self.get_temp_dir(),
                                 "ctgan_generator.keras")
        generator.save(save_path)
        reloaded_generator = keras.models.load_model(save_path)
        self.assertIsInstance(reloaded_generator, CTGANGenerator)
        outputs_reloaded = reloaded_generator(z)
        self.assertAllClose(ops.convert_to_numpy(outputs_original),
                            ops.convert_to_numpy(outputs_reloaded))
