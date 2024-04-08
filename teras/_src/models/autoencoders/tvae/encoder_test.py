import os

import keras
from keras import ops
from keras.src.testing.test_case import TestCase
from teras._src.preprocessing.data_samplers.tvae import TVAEDataSampler

from teras._src.models.autoencoders.tvae.encoder import TVAEEncoder
from teras._src.preprocessing.data_transformers.tvae import TVAEDataTransformer
from teras._src.utils import generate_fake_gemstone_data


class TVAEEncoderTest(TestCase):
    def setUp(self):
        fake_gem_df = generate_fake_gemstone_data(num_samples=16)
        cont_cols = ["depth", "table"]
        cat_cols = ["cut", "color", "clarity"]
        data_transformer = TVAEDataTransformer(continuous_features=cont_cols,
                                               categorical_features=cat_cols)
        x_transformed = data_transformer.fit_transform(fake_gem_df)
        self.batch_size = 8
        data_sampler = TVAEDataSampler(metadata=data_transformer.get_metadata(),
                                       categorical_features=cat_cols,
                                       continuous_features=cont_cols,
                                       batch_size=self.batch_size,
                                       )
        self.dataset = data_sampler.get_dataset(x_transformed=x_transformed,
                                                x_original=fake_gem_df)
        self.data_batch = next(iter(self.dataset))
        self.metadata = data_transformer.get_metadata()
        self.data_dim = data_sampler.data_dim
        self.latent_dim = 16

    def test_valid_call(self):
        encoder = TVAEEncoder(latent_dim=self.latent_dim)
        outputs = encoder(self.data_batch)

    def test_valid_output_shape(self):
        encoder = TVAEEncoder(latent_dim=self.latent_dim)
        outputs = encoder(self.data_batch)
        # encoder returns mean, log_var
        self.assertEqual(len(outputs), 2)
        # mean output shape check
        self.assertEqual(ops.shape(outputs[0]),
                         (self.batch_size, self.latent_dim))
        # log_var output shape check (yes both have the same shape)
        self.assertEqual(ops.shape(outputs[1]),
                         (self.batch_size, self.latent_dim))

    def test_model_save_and_load(self):
        encoder = TVAEEncoder(latent_dim=self.latent_dim)
        outputs = encoder(self.data_batch)
        save_path = os.path.join(self.get_temp_dir(),
                                 "tvae_encoder.keras"
                                 )
        encoder.save(save_path)
        reloaded_model = keras.models.load_model(save_path)

        # Check we got the real object back
        self.assertIsInstance(reloaded_model, TVAEEncoder)

        # Check that output matches
        reloaded_outputs = reloaded_model(self.data_batch)
        for original, reloaded in zip(outputs, reloaded_outputs):
            self.assertAllClose(
                ops.convert_to_numpy(original),
                ops.convert_to_numpy(reloaded)
            )
