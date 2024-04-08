import os
os.environ["KERAS_BACKEND"] = "jax"
import tensorflow as tf
from teras._src.models.gans.gain.generator import GAINGenerator
from teras._src.models.gans.gain.discriminator import GAINDiscriminator
import numpy as np
from teras._src.utils import inject_missing_values
from teras._src.backend.jax.trainers import gain as GAINTrainer


data_dim = 7
rng = np.random.default_rng(1337)
x_gen = rng.uniform(0., high=10., size=(128, data_dim)).astype(np.float32)
x_gen = inject_missing_values(x_gen)
x_disc = rng.uniform(0., high=10., size=(128, data_dim)).astype(np.float32)
x_disc = inject_missing_values(x_disc)
input_ds = tf.data.Dataset.from_tensor_slices((x_gen, x_disc))
input_ds = input_ds.batch(8)

generator = GAINGenerator(data_dim=data_dim)
discriminator = GAINDiscriminator(data_dim=data_dim)


GAINTrainer.init(generator, discriminator)
GAINTrainer.compile()

gen_state, disc_state = GAINTrainer.fit(input_ds, epochs=10)
