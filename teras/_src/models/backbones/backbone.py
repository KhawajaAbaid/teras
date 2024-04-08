import keras

from teras._src.api_export import teras_export


@teras_export("teras.models.Backbone")
class Backbone(keras.Model):
    """
    Base class for Backbone models.

    Copied from Keras-CV with some modifications.

    Reference(s):
    https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/backbones/backbone.py
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        config = {"name": self.name,
                  "trainable": self.trainable}
        return config

    @classmethod
    def from_config(cls, config):
        # The default `from_config()` for functional models will return a
        # vanilla `keras.Model`. We override it to get a subclass
        # instance back.
        return cls(**config)
