import keras
from teras.models.gans.pcgain import PCGAIN
from teras.models.gain.generator import GAINGenerator
from teras.models.gain.discriminator import GAINDiscriminator
from teras.models.pcgain.classifier import PCGAINClassifier
from keras import random, ops
from keras.src.testing.test_case import TestCase
from teras.utils.utils import inject_missing_values, create_gain_dataset
import numpy as np


class PCGAINTest(TestCase):
    def setUp(self):
        self.data_dim = 7
        self.rng = np.random.default_rng(1337)
        x_gen = self.rng.uniform(0., high=10., size=(8, self.data_dim)
                                 ).astype(np.float32)
        x_gen = inject_missing_values(x_gen)
        x_disc = self.rng.uniform(0., high=10., size=(8, self.data_dim)
                                  ).astype(np.float32)
        x_disc = inject_missing_values(x_disc)
        self.input_ds = create_gain_dataset(x_gen)
        if keras.backend.backend() == "torch":
            from torch.utils.data import DataLoader
            self.input_ds = DataLoader(self.input_ds, batch_size=8)
        else:
            self.input_ds = self.input_ds.batch(batch_size=8)

    def test_fit(self):
        generator = GAINGenerator(data_dim=self.data_dim)
        discriminator = GAINDiscriminator(data_dim=self.data_dim)
        classifier = PCGAINClassifier(num_classes=4,
                                      data_dim=self.data_dim)
        pcgain = PCGAIN(generator,
                        discriminator,
                        classifier)
        pcgain.compile(generator_optimizer=keras.optimizers.Adam(),
                       discriminator_optimizer=keras.optimizers.Adam())
        pcgain.build((8, self.data_dim))
        logs = pcgain.fit(self.input_ds)

    def test_predict(self):
        generator = GAINGenerator(data_dim=self.data_dim)
        discriminator = GAINDiscriminator(data_dim=self.data_dim)
        classifier = PCGAINClassifier(num_classes=4,
                                      data_dim=self.data_dim)
        pcgain = PCGAIN(generator,
                        discriminator,
                        classifier)
        pcgain.compile(generator_optimizer=keras.optimizers.Adam(),
                       discriminator_optimizer=keras.optimizers.Adam())
        pcgain.build((8, self.data_dim))
        logs = pcgain.fit(self.input_ds)
        x_test = self.rng.uniform(0., high=10., size=(8, self.data_dim))
        x_test = inject_missing_values(x_test)
        x_imputed = pcgain.predict(x_test)
        self.assertEqual(np.isnan(x_imputed).sum(), 0)
