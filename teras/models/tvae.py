import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, initializers
from tensorflow.keras import backend as K
from teras.losses.tvae import elbo_loss_tvae
from typing import Union, List, Tuple
from tqdm import tqdm


LIST_OR_TUPLE = Union[List[int], Tuple[int]]
LAYER_OR_MODEL = Union[layers.Layer, models.Model]


class Encoder(keras.Model):
    """
    Encoder for the TVAE model as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        latent_dim: `int`, default 128.
            Dimensionality of the learned latent space
        units_values: `List[int]` or `Tuple[int]`, default (128, 128),
            A list or tuple of integers.
            For each value in the sequence, a dense layer of that
            dimensions (units) is added to construct a compression block.
    """
    def __init__(self,
                 latent_dim: int = 128,
                 units_values: LIST_OR_TUPLE = (128, 128),
                 **kwargs):
        super().__init__(**kwargs)

        if not isinstance(units_values, (list, tuple)):
            raise ValueError(f"""`units_values` must be a list or tuple of units which determines
                        the number of compression layers and the dimensionality of those layers.
                        Received: {units_values}""")

        self.latent_dim = latent_dim
        self.units_values = units_values

        self.compression_block = models.Sequential(name="compression_block")
        for i, units in enumerate(self.units_values, start=1):
            self.compression_block.add(layers.Dense(units=units,
                                                    activation="relu",
                                                    name=f"compression_layer_{i}"))
        self.dense_mean = layers.Dense(units=self.latent_dim,
                                       name="mean")
        self.dense_log_var = layers.Dense(units=self.latent_dim,
                                          name="log_var")

    def call(self, inputs):
        h = self.compression_block(inputs)
        mean = self.dense_mean(h)
        log_var = self.dense_log_var(h)
        std = tf.exp(0.5 * log_var)
        return mean, std, log_var

    def get_config(self):
        config = super().get_config()
        new_config = {'latent_dim': self.latent_dim,
                      'units_values': self.units_values
                      }
        config.update(new_config)
        return config

class Decoder(keras.Model):
    """
    Encoder for the TVAE model as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        data_dim: `int`,
            Dimensionality of the input dataset. It is used to create the last layer for
            the decompression_block that projects data back to the original dimensions.
        units_values: `List[int]` or `Tuple[int]`,
            A list or tuple of integers. For each value in the sequence,
            a dense layer of that dimensionality is added to construct a decompression block.
            Note that, a dense layer of `data_dim` is also appended at the of decompression
            block to project data back to the original data dimensions.
    """
    def __init__(self,
                 data_dim: int = None,
                 units_values: LIST_OR_TUPLE = (128, 128),
                 **kwargs):
        super().__init__(**kwargs)

        if data_dim is None:
            raise ValueError("`data_dim` cannot be None, "
                             "as `data_dim` value is required to project data back to original data dimensions.")

        if not isinstance(units_values, (list, tuple)):
            raise ValueError(f"""`units_values` must be a list or tuple of units which determines
                        the number of decompression layers and the dimensionality of those layers.
                        Received: {units_values}""")

        self.data_dim = data_dim
        self.units_values = units_values

        self.decompression_block = models.Sequential(name="decompression_block")
        for i, units in enumerate(self.units_values, start=1):
            self.decompression_block.add(layers.Dense(units=units,
                                                      activation="relu",
                                                      name=f"decompression_layer_{i}"
                                                      )
                                         )
        self.decompression_block.add(layers.Dense(units=self.data_dim,
                                                  name="projection_to_data_dim"))

        self.sigmas = tf.Variable(initial_value=initializers.ones()(shape=(self.data_dim,)) * 0.1,
                                  trainable=True,
                                  name="sigmas")

    def call(self, inputs):
        x_generated = self.decompression_block(inputs)
        return x_generated, self.sigmas

    def get_config(self):
        config = super().get_config()
        new_config = {'data_dim': self.data_dim,
                      'units_values': self.units_values
                      }
        config.update(new_config)
        return config


# TODO: we should get rid of this data_dim parameter here since it's just the dimensionality
#   of the inputs received by the TVAE model
class TVAE(keras.Model):
    """
    TVAE model for tabular data generation
    based on the architecture proposed by the
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        data_dim: `int`,
            The dimensionality of the input dataset.
            It should be the dimensionality of the dataset that is passed to the `fit`
            method and not necessarily the dimensionality of the original input data
            as sometimes data transformations can alter the dimensionality of the data.
            Conveniently, if you're using DataSampler class, you can access `.data_dim`
            attribute to get the dimensionality of data.
        meta_data: `dict`,
            A dictionary containing metadata for all features in the input data.
            This metadata is computed during the data transformation step and can be accessed
            from `.get_meta_data()` method of DataTransformer instance.
        units_values: `List[int]` or `Tuple[int]`, default (128, 128),
            A list or tuple of units to construct the compression block in Encoder.
            For each value in the sequence, a dense layer of that
            dimensions (units) is added to construct a compression block.
        decoder_units_values: `List[int]` or `Tuple[int]`,
            A list or tuple of units to construct the decompression block in Decoder.
            For each value in the sequence,
            a dense layer of that dimensionality is added to construct a decompression block.
            Note that, a dense layer of `data_dim` is also appended at the of decompression
            block to project data back to the original data dimensions.
        latent_dim: `int`, default 128,
            Dimensionality of the learned latent space.
        loss_factor: `float`, default 2,
            Hyperparameter used in the computation of ELBO loss for TVAE.
            It controls how much the cross entropy loss contributes to the overall loss.
            It is directly proportional to the cross entropy loss.
    """
    def __init__(self,
                 data_dim: int,
                 meta_data: dict = None,
                 encoder_units_values: LIST_OR_TUPLE = (128, 128),
                 decoder_units_values: LIST_OR_TUPLE = (128, 128),
                 latent_dim: int = 128,
                 loss_factor: float = 2,
                 **kwargs):
        super().__init__(**kwargs)

        if meta_data is None:
            raise ValueError("`meta_data` cannot be None. "
                             "You can access it through `.get_meta_data()` method of the DataTransformer instance.")

        self.data_dim = data_dim
        self.meta_data = meta_data
        self.encoder_units_values = encoder_units_values
        self.decoder_units_values = decoder_units_values
        self.latent_dim = latent_dim
        self.loss_factor = loss_factor

        self.encoder = Encoder(latent_dim=self.latent_dim,
                               units_values=self.encoder_units_values)
        self.decoder = Decoder(data_dim=self.data_dim,
                               units_values=self.decoder_units_values)

    def call(self, inputs, training=None):
        mean, log_var, std = self.encoder(inputs)
        eps = tf.random.uniform(minval=0, maxval=1,
                                shape=tf.shape(std),
                                dtype=std.dtype)
        z = (std * eps) + mean
        generated_samples, sigmas = self.decoder(z)
        if training:
            loss = elbo_loss_tvae(data_dim=self.data_dim,
                                  real_samples=inputs,
                                  generated_samples=generated_samples,
                                  meta_data=self.meta_data,
                                  sigmas=sigmas,
                                  mean=mean,
                                  log_var=log_var,
                                  loss_factor=self.loss_factor)
            self.add_loss(loss)
            updated_simgas = tf.clip_by_value(self.decoder.sigmas,
                                              clip_value_min=0.01,
                                              clip_value_max=1.0)
            K.update(self.decoder.sigmas, updated_simgas)
        return generated_samples

    def generate(self,
                 num_samples: int,
                 data_transformer=None,
                 reverse_transform: bool = False,
                 batch_size: int = 512):
        """
        Generates new data using the trained Generator.

        Args:
            num_samples: `int`,
                Number of new samples to generate
            data_transformer:
                Instance of DataTransformer class used to preprocess the raw data.
                This is required only if the `reverse_transform` is set to True.
            reverse_transform: bool, default False,
                Whether to reverse transform the generated data to the original data format.
                If False, the raw generated data will be returned, which you can then manually
                transform into original data format by utilizing DataTransformer instance's
                `reverse_transform` method.
            batch_size: int, default 512.
                If a value is passed, samples will be generated in batches
                where `batch_size` determines the size of each batch.
                If `None`, all `num_samples` will be generated at once.
                Note that, if the number of samples to generate aren't huge
                or you know your hardware can handle to generate all samples at once,
                you can leave the value to None, otherwise it is recommended to specify
                a value for batch size.
        """
        if batch_size is None:
            batch_size = num_samples
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
            if data_transformer is None:
                raise ValueError("""To reverse transform the raw generated data, `data_transformer` must not be None.
                             Please pass the instance of DataTransformer class used to transform the input
                             data. Or alternatively, you can set `reverse_transform` to False, and later
                             manually reverse transform the generated raw data to original format.""")
            generated_samples = data_transformer.reverse_transform(x_generated=generated_samples)

        return generated_samples

    def get_config(self):
        config = super().get_config()
        new_config = {'data_dim': self.data_dim,
                      'meta_data': self.meta_data,
                      'encoder_units_values': self.encoder_units_values,
                      'decoder_units_values': self.decoder_units_values,
                      'latent_dim': self.latent_dim,
                      'loss_factor': self.loss_factor,
                      }
        config.update(new_config)
        return config

