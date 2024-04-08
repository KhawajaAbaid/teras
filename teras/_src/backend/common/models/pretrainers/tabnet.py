import keras
from keras import random, ops
from keras.backend import floatx


class BaseTabNetPretrainer(keras.Model):
    def __init__(self,
                 encoder: keras.Model,
                 decoder: keras.Model,
                 missing_feature_probability: float = 0.3,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.missing_feature_probability = missing_feature_probability
        self.seed = seed

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self._seed_generator = random.SeedGenerator(seed=self.seed)
        self._pretrained = False

    def build(self, input_shape):
        self.encoder.build(input_shape)
        input_shape = self.encoder.compute_output_shape(input_shape)
        self.decoder.build(input_shape)

    def compute_loss(self, x, x_reconstructed, mask):
        nominator_part = (x_reconstructed - x) * mask
        real_samples_population_std = ops.std(ops.cast(x, dtype=floatx()))
        # divide
        x = nominator_part / real_samples_population_std
        # Calculate L2 norm
        loss = ops.sqrt(ops.sum(ops.square(x)))
        return loss

    def reset_metrics(self):
        self.loss_tracker.reset_state()

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, mask, **kwargs):
        x = inputs * (1 - mask)
        # Encoded representations
        x = self.encoder(x, mask=(1 - mask))
        # Reconstructed features
        x = self.decoder(x, mask=mask)
        self._pretrained = True
        return x

    def get_config(self):
        config = {
            "name": self.name,
            "trainable": self.trainable,
            "encoder": keras.layers.serialize(self.encoder),
            "decoder": keras.layers.serialize(self.decoder),
            "missing_feature_probability":
                self.missing_feature_probability,
            "seed": self.seed,
        }
        return config

    @classmethod
    def from_config(cls, config):
        encoder = keras.layers.deserialize(config.pop("encoder"))
        decoder = keras.layers.deserialize(config.pop("decoder"))
        return cls(encoder=encoder, decoder=decoder, **config)

    @property
    def pretrained_encoder(self):
        if not self._pretrained:
            raise AssertionError(
                "The `fit` method of the `TabNetPretrainer` has not yet "
                "been called. Please train the `TabNetPretrainer` before "
                "accessing the `pretrained_encoder` attribute."
            )
        return self.encoder

    @property
    def pretrained_decoder(self):
        if not self._pretrained:
            raise AssertionError(
                "The `fit` method of the `TabNetPretrainer` has not yet "
                "been called. Please train the `TabNetPretrainer` before "
                "accessing the `pretrained_decoder` attribute."
            )
        return self.encoder
