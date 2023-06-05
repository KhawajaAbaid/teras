import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from teras.preprocessing.ctgan import DataTransformer
from teras.losses.tvae import TvaeElboLoss
from teras.layers.tvae import Encoder, Decoder
from typing import Union, List, Tuple
from tqdm import tqdm


LIST_OR_TUPLE = Union[List[int], Tuple[int]]


class TVAE(keras.Model):
    """
    TVAE model based on the architecure proposed by the
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        latent_dim: Dimensionality of the learned latent space
            Defaults to 128
        compress_dims: A list or tuple of integers. For each value in the sequence,
            a (dense) compression layer is added.
            Defaults to (128, 128)
        compress_dims: A list or tuple of integers. For each value in the sequence,
            a (dense) decompression layer is added.
            Defaults to (128, 128)
        data_transformer: An instance of DataTransformer class.
        loss_factor: Hyperparameter used in the computation of TVAE ELBO loss.
            Defaults to 2.
    """
    def __init__(self,
                 latent_dim: int = 128,
                 compress_dims: LIST_OR_TUPLE = (128, 128),
                 decompress_dims: LIST_OR_TUPLE = (128, 128),
                 data_transformer: DataTransformer = None,
                 loss_factor=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.data_transformer = data_transformer
        self.loss_factor = loss_factor

        self.features_meta_data = self.data_transformer.features_meta_data
        self.data_dim = self.features_meta_data["total_transformed_features"]

        self.encoder = Encoder(latent_dim=self.latent_dim,
                               compress_dims=self.compress_dims)
        self.decoder = Decoder(data_dim=self.data_dim,
                               decompress_dims=self.decompress_dims)
        self.tvae_elbo_loss = TvaeElboLoss(features_meta_data=self.features_meta_data)

    def call(self, x_real, training=None):
        mean, log_var, std = self.encoder(x_real)
        eps = tf.random.uniform(minval=0, maxval=1,
                                shape=tf.shape(std),
                                dtype=std.dtype)
        z = (std * eps) + mean
        x_generated, sigmas = self.decoder(z)
        if training:
            loss = self.tvae_elbo_loss(x_real,
                                       x_generated=x_generated,
                                       sigmas=sigmas,
                                       mean=mean,
                                       log_var=log_var,
                                       loss_factor=self.loss_factor)
            self.add_loss(loss)
            updated_simgas = tf.clip_by_value(self.decoder.sigmas,
                                                   clip_value_min=0.01,
                                                   clip_value_max=1.0)
            K.update(self.decoder.sigmas, updated_simgas)
        return x_generated

    def generate_new_data(self,
                          num_samples,
                          reverse_transform=True,
                          batch_size=512):
        """
        Generates new data using the trained Generator.

        Args:
            num_samples: Number of new samples to generate
            reverse_transform: Whether to reverse transform the generated data to the original data format.
                Defaults to True. If False, the raw generated data will be returned.
            batch_size: Specify the batch size to generate new samples.
                Defaults to 512.
        """
        num_steps = num_samples // batch_size
        num_steps += 1 if num_samples % batch_size != 0 else 0
        generated_samples = []
        for _ in tqdm(range(num_steps), desc="Generating Data"):
            z = tf.random.normal(shape=[batch_size, self.latent_dim])
            gen_samples_temp, _ = self.decoder(z)
            generated_samples.append(gen_samples_temp)
        generated_samples = tf.concat(generated_samples, axis=0)
        generated_samples = generated_samples[:num_samples]

        if reverse_transform:
            generated_samples = self.data_transformer.reverse_transform(x_generated=generated_samples)

        return generated_samples
