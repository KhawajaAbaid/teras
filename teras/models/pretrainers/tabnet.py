import keras
from teras.losses.tabnet import tabnet_reconstruction_loss


class BaseTabNetPretrainer(keras.Model):
    """
    Base pretrainer model class for tabnet.

    Args:
        encoder: keras.Model, an instance of `TabNetEncoder`
        decoder: keras.Model, an instance of `TabNetDecoder`
        missing_feature_probability: float, probability of missing features
    """
    def __init__(self,
                 encoder: keras.Model,
                 decoder: keras.Model,
                 missing_feature_probability: float = 0.3,
                 **kwargs):
        super().__init__(base_model=encoder,
                         **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.missing_feature_probability = missing_feature_probability

        self._reconstruction_loss_fn = None
        self._reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

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
        _metrics = super().metrics()
        _metrics.append(self._reconstruction_loss_tracker)
        return _metrics

    def call(self, inputs, mask):
        inputs *= (1 - mask)
        # Encoded representations
        x = self.encoder(inputs, mask=(1 - mask))
        # Reconstructed features
        x = self.decoder(x, mask=mask)
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
