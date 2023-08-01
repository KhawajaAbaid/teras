import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from teras.losses.tvae import elbo_loss_tvae
from teras.utils.types import LayerOrModelType
from tqdm import tqdm


@keras.saving.register_keras_serializable(package="teras.layerflow.models")
class TVAE(keras.Model):
    """
    TVAE model with LayerFlow design.
    TVAE is a tabular data generation architecture proposed by the
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

        encoder: ``keras.layers.Layer`` or ``keras.models.Model``,
            An instance of ``TVAEEncoder`` or any custom keras layer/model that can work
            in its place.
            You can import the ``TVAEEncoder`` as follows,
                >>> from teras.models import TVAEEncoder

        decoder: ``keras.layers.Layer`` or ``keras.models.Model``,
            An instance of ``TVAEDecoder`` or any custom keras model/layer that can work
            in its palce.
            It decodes or decompresses from latent dimensions to data dimensions.
            You can import the ``TVAEDecoder`` as follows,
                >>> from teras.models import TVAEDecoder

        latent_dim: ``int``, default 128,
            Dimensionality of the learned latent space

        loss_factor: ``float``, default 2.,
            Hyperparameter used in the computation of ``ELBO loss`` for ``TVAE``.
            It controls how much the cross entropy loss contributes to the overall loss.
            It is directly proportional to the cross entropy loss.
    """
    def __init__(self,
                 encoder: LayerOrModelType,
                 decoder: LayerOrModelType,
                 latent_dim: int = 128,
                 loss_factor: float = 2.,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.loss_factor = loss_factor

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
            num_samples: ``int``,
                Number of new samples to generate

            data_transformer:
                Instance of ``TVAEDataTransformer`` class used to preprocess the raw data.
                This is required only if the `reverse_transform` is set to True.

            reverse_transform: ``bool``, default False,
                Whether to reverse transform the generated data to the original data format.
                If False, the raw generated data will be returned, which you can then manually
                transform into original data format by utilizing ``TVAEDataTransformer`` instance's
                ``reverse_transform`` method.

            batch_size: ``int``, default 512.
                If a value is passed, samples will be generated in batches
                where ``batch_size`` determines the size of each batch.
                If ``None``, all `num_samples` will be generated at once.
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
        config.update({'encoder': keras.layers.serialize(self.encoder),
                       'decoder': keras.layers.serialize(self.decoder),
                       'latent_dim': self.latent_dim,
                       'loss_factor': self.loss_factor,
                       })
        return config

    @classmethod
    def from_config(cls, config):
        encoder = keras.layers.deserialize(config.pop("encoder"))
        decoder = keras.layers.deserialize(config.pop("decoder"))
        return cls(encoder=encoder,
                   decoder=decoder,
                   **config)
