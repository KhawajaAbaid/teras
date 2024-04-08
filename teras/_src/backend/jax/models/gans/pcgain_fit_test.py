import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import tensorflow as tf
from teras._src.backend.jax.models.gans.pcgain import PCGAIN
from teras._src.models.gans.gain.generator import GAINGenerator
from teras._src.models.gans.gain.discriminator import GAINDiscriminator
from teras._src.models.gans.pcgain.classifier import PCGAINClassifier
import numpy as np
from teras._src.utils import inject_missing_values


DATA_DIM = 7

NUM_SAMPLES = 256
BATCH_SIZE = 32

rng = np.random.default_rng(1337)
x_gen = rng.uniform(0., high=10., size=(NUM_SAMPLES, DATA_DIM)).astype(np.float32)
x_gen = inject_missing_values(x_gen)
x_disc = rng.uniform(0., high=10., size=(NUM_SAMPLES, DATA_DIM)).astype(np.float32)
x_disc = inject_missing_values(x_disc)
input_ds = tf.data.Dataset.from_tensor_slices((x_gen, x_disc))
input_ds = input_ds.batch(BATCH_SIZE)

generator = GAINGenerator(data_dim=DATA_DIM)
discriminator = GAINDiscriminator(data_dim=DATA_DIM)
classifier = PCGAINClassifier(num_classes=4,
                              data_dim=DATA_DIM)
pcgain = PCGAIN(generator,
                discriminator,
                classifier)
pcgain.build(tuple(x_gen.shape))
pcgain.compile(generator_optimizer=keras.optimizers.Adam(),
               discriminator_optimizer=keras.optimizers.Adam())
pcgain.build_optimizers()
history = pcgain.fit(input_ds, epochs=10)
