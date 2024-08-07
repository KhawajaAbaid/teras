import keras
from keras import random, ops
from teras._src.utils import clean_reloaded_config_data
from teras._src.decorators import assert_fitted


class BaseTVAE(keras.Model):
    """
    Base TVAE class.
    """
    def __init__(self,
                 encoder: keras.Model,
                 decoder: keras.Model,
                 metadata: dict,
                 data_dim: int,
                 latent_dim: int = 128,
                 loss_factor: float = 2.,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.metadata = metadata
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.loss_factor = loss_factor
        self.seed = seed
        self._seed_gen = random.SeedGenerator(self.seed)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self._input_shape = input_shape
        if not self.encoder.built:
            self.encoder.build(input_shape)
        input_shape = input_shape[:-1] + (self.latent_dim,)
        if not self.decoder.built:
            self.decoder.build(input_shape)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compute_loss(self, real_samples=None, generated_samples=None,
                     sigmas=None, mean=None, log_var=None):
        """
        Evidence Lower Bound (ELBO) Loss [1] adapted for
        TVAE architecture [2] proposed by Lei Xu et al. in the paper,
        "Modeling Tabular data using Conditional GAN".

        Reference(s):
            [1]: https://arxiv.org/abs/1312.6114
            [2]: https://arxiv.org/abs/1907.00503

        Args:
            real_samples: Samples drawn from the input dataset.
            generated_samples: Samples generated by the decoder.
            sigmas: Sigma values returned by decoder
            mean: Mean values returned by encoder
            log_var: Log var values returned by encoder

        Returns:
            Elbo loss adapted for the `TVAE` model.
        """
        loss = []
        cross_entropy = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="sum")
        cont_i = 0  # continuous index
        cat_i = 0  # categorical index
        for i, relative_index in enumerate(self.metadata["relative_indices_all"]):
            # the first k features are continuous
            if i < self.metadata["num_continuous"]:
                # each continuous feature is of the form
                # [alpha, beta1, beta2...beta(n)] where n is the number of
                # clusters

                # calculate alpha loss
                std = sigmas[relative_index]
                eq = real_samples[:, relative_index] - ops.tanh(
                    generated_samples[:, relative_index])
                loss_temp = ops.sum((eq ** 2 / 2 / (std ** 2)))
                loss.append(loss_temp)
                loss_temp = ops.log(std) * ops.cast(
                    ops.shape(real_samples)[0], dtype=std.dtype)
                loss.append(loss_temp)

                # calculate betas loss
                num_clusters = self.metadata["continuous"][
                    "num_valid_clusters_all"][cont_i]
                logits = generated_samples[:, relative_index + 1: relative_index + 1 + num_clusters]
                labels = ops.argmax(
                    real_samples[:, relative_index + 1: relative_index + 1 + num_clusters],
                    axis=-1)
                cross_entropy_loss = cross_entropy(y_pred=logits, y_true=labels)
                loss.append(cross_entropy_loss)
                cont_i += 1
            else:
                num_categories = self.metadata["categorical"][
                    "num_categories_all"][cat_i]
                logits = generated_samples[:, relative_index: relative_index + num_categories]
                labels = ops.argmax(
                    real_samples[:, relative_index: relative_index + num_categories],
                    axis=-1)
                cross_entropy_loss = cross_entropy(y_pred=logits, y_true=labels)
                loss.append(cross_entropy_loss)
                cat_i += 1
        KLD = -0.5 * ops.sum(1 + log_var - mean ** 2 - ops.exp(log_var))
        loss_1 = ops.sum(loss) * self.loss_factor / self.data_dim
        loss_2 = KLD / self.data_dim
        final_loss = loss_1 + loss_2
        return final_loss

    def call(self, inputs):
        mean, log_var = self.encoder(inputs)
        std = ops.exp(0.5 * log_var)
        eps = random.uniform(shape=ops.shape(std),
                             minval=0, maxval=1,
                             dtype=std.dtype,
                             seed=self._seed_gen)
        z = (std * eps) + mean
        generated_samples, sigmas = self.decoder(z)
        return generated_samples, sigmas, mean, log_var

    def test_step(self, data):
        generated_samples, sigmas, mean, log_var = self(data)
        loss = self.compute_loss(real_samples=data,
                                 generated_samples=generated_samples,
                                 sigmas=sigmas,
                                 mean=mean,
                                 log_var=log_var)
        self.loss_tracker.update_state(loss)
        logs = {m.name: m.result() for m in self.metrics}
        return logs

    def predict_step(self, z):
        generated_samples, _ = self.decoder(z)
        return generated_samples

    def compute_output_shape(self, input_shape):
        batch_size, _ = input_shape
        return ((batch_size, self.data_dim),
                (batch_size, self.latent_dim),
                (self.data_dim,),
                (batch_size, self.latent_dim),
                (batch_size, self.latent_dim))

    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder': keras.layers.serialize(self.encoder),
            'decoder': keras.layers.serialize(self.decoder),
            'metadata': self.metadata,
            'data_dim': self.data_dim,
            'latent_dim': self.latent_dim,
            'loss_factor': self.loss_factor,
            'seed': self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder = keras.layers.deserialize(config.pop("encoder"))
        decoder = keras.layers.deserialize(config.pop("decoder"))
        metadata = clean_reloaded_config_data(config.pop("metadata"))
        return cls(encoder=encoder,
                   decoder=decoder,
                   metadata=metadata,
                   **config)

    def get_build_config(self):
        build_config = dict(input_shape=self._input_shape)
        return build_config

    def build_from_config(self, build_config):
        self.build(build_config["input_shape"])
