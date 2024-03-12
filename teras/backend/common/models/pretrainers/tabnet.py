import keras
from keras import random
from teras.losses.tabnet import tabnet_reconstruction_loss


class BaseTabNetPretrainer(keras.Model):
    def __init__(self,
                 encoder: keras.Model,
                 decoder: keras.Model,
                 missing_feature_probability: float = 0.3,
                 mask_seed: int = 1337,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.missing_feature_probability = missing_feature_probability
        self.mask_seed = mask_seed

        self._reconstruction_loss_fn = None
        self._reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self._mask_seed_generator = random.SeedGenerator(seed=self.mask_seed)
        self._pretrained = False

    def build(self, input_shape):
        self.encoder.build(input_shape)
        input_shape = self.encoder.compute_output_shape(input_shape)
        self.decoder.build(input_shape)

    def compile(self,
                loss=None,
                optimizer=None,
                reconstruction_loss=tabnet_reconstruction_loss,
                **kwargs
                ):
        super().compile(loss=loss, optimizer=optimizer, **kwargs)
        self._reconstruction_loss_fn = reconstruction_loss

    @property
    def metrics(self):
        _metrics = super().metrics
        _metrics.append(self._reconstruction_loss_tracker)
        return _metrics

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
