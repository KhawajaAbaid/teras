import tensorflow as tf
from tensorflow import keras
from teras.layerflow.models.tvae import TVAE as _TVAE_LF
from teras.utils.types import UnitsValuesType


@keras.saving.register_keras_serializable(package="teras.models")
class TVAEEncoder(keras.Model):
    """
    Encoder for the TVAE model as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        latent_dim: ``int``, default 128.
            Dimensionality of the learned latent space

        units_values: ``List[int]`` or ``Tuple[int]``, default (128, 128),
            A list or tuple of integers.
            For each value in the sequence, a dense layer of that
            dimensions (units) is added to construct a compression block.
    """
    def __init__(self,
                 latent_dim: int = 128,
                 units_values: UnitsValuesType = (128, 128),
                 **kwargs):
        super().__init__(**kwargs)

        if not isinstance(units_values, (list, tuple)):
            raise ValueError(f"""`units_values` must be a list or tuple of units which determines
                        the number of compression layers and the dimensionality of those layers.
                        Received: {units_values}""")

        self.latent_dim = latent_dim
        self.units_values = units_values

        self.compression_block = keras.models.Sequential(name="compression_block")
        for i, units in enumerate(self.units_values, start=1):
            self.compression_block.add(keras.layers.Dense(units=units,
                                                          activation="relu",
                                                          name=f"compression_layer_{i}"))
        self.dense_mean = keras.layers.Dense(units=self.latent_dim,
                                             name="mean")
        self.dense_log_var = keras.layers.Dense(units=self.latent_dim,
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


@keras.saving.register_keras_serializable(package="teras.models")
class TVAEDecoder(keras.Model):
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
                 units_values: UnitsValuesType = (128, 128),
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

        self.decompression_block = keras.models.Sequential(name="decompression_block")
        for i, units in enumerate(self.units_values, start=1):
            self.decompression_block.add(keras.layers.Dense(units=units,
                                                            activation="relu",
                                                            name=f"decompression_layer_{i}"
                                                            )
                                         )
        self.decompression_block.add(keras.layers.Dense(units=self.data_dim,
                                                        name="projection_to_data_dim"))

        self.sigmas = tf.Variable(initial_value=keras.initializers.ones()(shape=(self.data_dim,)) * 0.1,
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


@keras.saving.register_keras_serializable(package="teras.models")
class TVAE(_TVAE_LF):
    """
    TVAE model for tabular data generation
    based on the architecture proposed by the
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        input_dim: ``int``, dimensionality of the input dataset.
            The dimensionality of the input dataset.

            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.

            You can access the dimensionality of the transformed dataset through the
            ``.data_dim`` attribute of the ``TVAEDataSampler`` instance used in sampling
            the dataset.

        metadata: ``dict``,
            A dictionary containing metadata for all features in the input data.
            This metadata is computed during the data transformation step and can be accessed
            from ``.get_metadata()`` method of ``TVAEDataTransformer`` instance.

        units_values: ``List[int]`` or ``Tuple[int]``, default (128, 128),
            A list or tuple of units to construct the compression block in ``TVAEEncoder``.
            For each value in the sequence, a dense layer of that
            dimensions (units) is added to construct a compression block.

        decoder_units_values: ``List[int]`` or ``Tuple[int]``,
            A list or tuple of units to construct the decompression block in ``TVAEDecoder``.
            For each value in the sequence,
            a dense layer of that dimensionality is added to construct a decompression block.
            Note that, a dense layer of ``data_dim`` is also appended at the of decompression
            block to project data back to the original data dimensions.

        latent_dim: `int`, default 128,
            Dimensionality of the learned latent space.

        loss_factor: `float`, default 2,
            Hyperparameter used in the computation of ELBO loss for TVAE.
            It controls how much the cross entropy loss contributes to the overall loss.
            It is directly proportional to the cross entropy loss.
    """
    def __init__(self,
                 input_dim: int,
                 metadata: dict = None,
                 encoder_units_values: UnitsValuesType = (128, 128),
                 decoder_units_values: UnitsValuesType = (128, 128),
                 latent_dim: int = 128,
                 loss_factor: float = 2,
                 **kwargs):
        if metadata is None:
            raise ValueError("`metadata` cannot be None. "
                             "You can access it through `.get_metadata()` method of the `TVAEDataTransformer` instance.")
        encoder = TVAEEncoder(latent_dim=latent_dim,
                              units_values=encoder_units_values)
        decoder = TVAEDecoder(data_dim=input_dim,
                              units_values=decoder_units_values)
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         latent_dim=latent_dim,
                         loss_factor=loss_factor,
                         **kwargs)
        self.input_dim = input_dim
        self.metadata = metadata
        self.encoder_units_values = encoder_units_values
        self.decoder_units_values = decoder_units_values
        self.latent_dim = latent_dim
        self.loss_factor = loss_factor

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'input_dim': self.input_dim,
                  'metadata': self.metadata,
                  'encoder_units_values': self.encoder_units_values,
                  'decoder_units_values': self.decoder_units_values,
                  'latent_dim': self.latent_dim,
                  'loss_factor': self.loss_factor,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        return cls(input_dim=input_dim,
                   **config)
