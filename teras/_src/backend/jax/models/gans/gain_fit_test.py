import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import tensorflow as tf
from teras._src.backend.jax.models.gans.gain import GAIN
from teras._src.models.gans.gain.generator import GAINGenerator
from teras._src.models.gans.gain.discriminator import GAINDiscriminator
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
gain = GAIN(generator,
            discriminator)
gain.build(tuple(x_gen.shape))
gain.compile(generator_optimizer=keras.optimizers.Adam(),
             discriminator_optimizer=keras.optimizers.Adam())
gain.build_optimizers()
history = gain.fit(input_ds, epochs=10)
