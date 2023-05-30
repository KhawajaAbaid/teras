import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from teras.preprocessing.ctgan import DataTransformer
from teras.losses.tvae import TvaeElboLoss
from teras.layers.tvae import Encoder, Decoder
from typing import Union, List, Tuple


LIST_OR_TUPLE = Union[List[int], Tuple[int]]


class TVAE(keras.Model):
    def __init__(self,
                 latent_dim: int = 128,
                 compress_dims: LIST_OR_TUPLE = (128, 128),
                 decompress_dims: LIST_OR_TUPLE = (128, 128),
                 data_transformer: DataTransformer = None,
                 l2_scale=1e-5,
                 loss_factor=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.data_transformer = data_transformer
        self.l2_scale = l2_scale
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