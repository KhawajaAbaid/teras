import keras
from teras.models.gans.ctgan import CTGAN
from teras.models.gans.ctgan.generator import CTGANGenerator
from teras.models.gans.ctgan.discriminator import CTGANDiscriminator
from teras.preprocessing.data_transformers.ctgan import CTGANDataTransformer
from teras.preprocessing.data_samplers.ctgan import CTGANDataSampler
from keras.src.testing.test_case import TestCase
from teras.utils.utils import generate_fake_gemstone_data


class CTGANTest(TestCase):
    def setUp(self):
        fake_gem_df = generate_fake_gemstone_data(num_samples=256)
        cont_cols = ["depth", "table"]
        cat_cols = ["cut", "color", "clarity"]
        data_transformer = CTGANDataTransformer(continuous_features=cont_cols,
                                                categorical_features=cat_cols)
        x_transformed = data_transformer.fit_transform(fake_gem_df)

        data_sampler = CTGANDataSampler(metadata=data_transformer.get_metadata(),
                                        categorical_features=cat_cols,
                                        continuous_features=cont_cols,
                                        batch_size=32,
                                        )
        self.dataset = data_sampler.get_dataset(x_transformed=x_transformed,
                                                x_original=fake_gem_df)
        self.metadata = data_transformer.get_metadata()
        self.data_dim = data_sampler.data_dim
        # if keras.backend.backend() == "torch":
        #     from torch.utils.data import DataLoader
        #     self.input_ds = DataLoader(self.input_ds, batch_size=8)
        # else:
        #     self.input_ds = self.input_ds.batch(batch_size=8)

    def test_fit(self):
        generator = CTGANGenerator(data_dim=self.data_dim,
                                   metadata=self.metadata)
        discriminator = CTGANDiscriminator()
        ctgan = CTGAN(generator=generator,
                      discriminator=discriminator,
                      metadata=self.metadata)
        ctgan.compile(generator_optimizer=keras.optimizers.Adam(),
                      discriminator_optimizer=keras.optimizers.Adam())
        ctgan.build((16, self.data_dim))
        logs = ctgan.fit(self.dataset, epochs=3)
