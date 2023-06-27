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
        latent_dim: `int`, default 128. Dimensionality of the learned latent space
        units_values: default (128, 128), A list or tuple of integers.
            For each value in the sequence, a (dense) compression layer of that
            dimensions (units) is added to construct a compression block.
        compression_block: `layers.Layer` or `models.Model`, Serves as the compression block.
            This parameter gives you full control over the compression block architecture.
            If specified, the `units_values` will be ignored.
    """
    def __init__(self,
                 latent_dim: int = 128,
                 units_values: LIST_OR_TUPLE = (128, 128),
                 compression_block: LAYER_OR_MODEL = None,
                 **kwargs):
        super().__init__(**kwargs)

        if compression_block is None and units_values is None:
            raise ValueError(f"""`units_values` and `compression_block` both cannot be None at the same time. 
                    You must specify one of them.
                    Received, `units_values`: {units_values}, `compression_block`: {compression_block}""")

        self.latent_dim = latent_dim
        self.units_values = units_values
        self.compression_block = compression_block

        if compression_block is None:
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


class Decoder(keras.Model):
    """
    Encoder for the TVAE model as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        data_dim: `int`, Dimensionality of the data.
            This parameter is must unless you pass your own decompression_block,
            in which case you can leave this as `None`.
        units_values: A list or tuple of integers. For each value in the sequence,
            a (dense) decompression layer is added to construct a decompression block.
            Note that, a dense layer of `data_dim` is appended at the end by model
            to project data back to the original data dimensions.
        decompression_block: `layers.Layer` or `models.Model`, decompresses the compressed
            latent dimension data back to original data dimensions.
            Note that, if you specify a custom decompression block, the last layer must
            project the data back to original dimensions i.e. `data_dim`.
    """
    def __init__(self,
                 data_dim: int = None,
                 units_values: LIST_OR_TUPLE = (128, 128),
                 decompression_block: LAYER_OR_MODEL = None,
                 **kwargs):
        super().__init__(**kwargs)

        if decompression_block is None and units_values is None:
            raise ValueError(f"""`units_values` and `decompression_block` both cannot be None at the same time. 
                    You must specify one of them.
                    Received, `units_values`: {units_values}, `decompression_block`: {decompression_block}""")

        if decompression_block is None and units_values is not None and data_dim is None:
            raise ValueError(f"""`data_dim` cannot be None unless you specify your own `decompression_block`, 
                    as `data_dim` value is required to project data back to original data dimensions.
                    Please either specify `data_dim` or alternatively you can pass your own custom 
                    decompression block, in which case you can leave `data_dim` as None. """)

        self.data_dim = data_dim
        self.units_values = units_values
        self.decompression_block = decompression_block

        if self.decompression_block is None:
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


class TVAE(keras.Model):
    """
    TVAE model for tabular data generation
    based on the architecture proposed by the
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        encoder: `layers.Layer` or `models.Model`. It encodes or compresses
            data useful hidden dimensional representations.
            If `None`, a default encoder is constructed with (128, 128)
            dimensions is constructed.
            You can import the Encoder from,
            `from teras.models.tvae import Encoder`, customize it using available
            parameters, subclass it or construct your own Encoder from scratch and
            pass it here.
        decoder: `layers.Layer` or `models.Model`. It decodes or decompresses
            from latent dimensions to data dimensions.
            If `None`, a default decoder is constructed with (128, 128) hidden
            dimensions along with a dense output layer that projects the data
            back to original dimensions.
            You can import the Decoder from,
            `from teras.models.tvae import Decoder`, customize it using available
            parameters, subclass it or construct your own Encoder from scratch and
            pass it here.
        latent_dim: `int`, default 128, Dimensionality of the learned latent space
        data_dim: `int`, dimensionality of the input dataset.
            It should be the dimensionality of the dataset that is passed to the `fit`
            method and not necessarily the dimensionality of the original input data
            as sometimes data transformations can alter the dimensionality of the data.
            Conveniently, if you're using DataSampler class, you can access `.data_dim`
            attribute to get the dimensionality of data.
        loss_factor: default 2, hyperparameter used in the computation of ELBO loss for TVAE.
            It controls how much the cross entropy loss contributes to the overall loss.
            It is directly proportional to the cross entropy loss.
    """
    def __init__(self,
                 encoder: LAYER_OR_MODEL = None,
                 decoder: LAYER_OR_MODEL = None,
                 latent_dim: int = 128,
                 data_dim: int = None,
                 loss_factor=2,
                 meta_data=None,
                 **kwargs):
        super().__init__(**kwargs)

        if data_dim is None and (encoder is None and decoder is None):
            raise ValueError(f"""`data_dim` is required to instantiate the Encoder and Decoder objects.
                    But {data_dim} was passed.
                    You can either pass the value for `data_dim` -- which can be accessed through `.data_dim`
                    attribute of DataSampler instance if you don't know the data dimensions --
                    or alternatively you can instantiate and pass your own Encoder and Decoder instances,
                    either by importing them `from teras.models.tvae import Encoder, Decoder` or constructing
                    your own from scratch, in which case you can leave the `data_dim` parameter as None.""")

        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder
        self.data_dim = data_dim
        self.loss_factor = loss_factor
        self.meta_data = meta_data

        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.decoder = Decoder(data_dim=self.data_dim)

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
                 num_samples,
                 data_transformer=None,
                 reverse_transform=True,
                 batch_size=512):
        """
        Generates new data using the trained Generator.

        Args:
            num_samples: Number of new samples to generate
            data_transformer: Instance of DataTransformer class used to preprocess
                the raw data.
                This is required only if the `reverse_transform` is set to True.
            reverse_transform: bool, default True,
                whether to reverse transform the generated data to the original data format.
                If False, the raw generated data will be returned, which you can then manually
                transform into original data format by utilizing DataTransformer instance's
                `reverse_transform` method.
            batch_size: int, default None.
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
