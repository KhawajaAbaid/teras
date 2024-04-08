import keras


class Pretrainer(keras.Model):
    """Base Pretrainer class for the architectures that make use of
    semi-supervised pretraining.

    Args:
        base_model: keras.Model instance, that can be a base model or
            part of the architecture like an encoder in an encoder
            decoder model where only the encoder needs to be pretrained.
    """
    def __init__(self,
                 base_model,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self._pretrained = False
        self._pretrained_model = None

    @property
    def pretrained_model(self):
        return self._pretrained_model

    @pretrained_model.setter
    def pretrained_model(self, value):
        if not isinstance(value, keras.Model):
            raise ValueError(
                "Pretrained model must be an instance of `keras.Model`. "
                f"Received: {type(value)}.")
        self._pretrained_model = value

    @pretrained_model.getter
    def pretrained_model(self):
        if not self._pretrained:
            raise Exception("Model has not yet been pretrained.")
        return self._pretrained_model

    def is_pretrained(self):
        return self._pretrained

    def get_config(self):
        config = super().get_config()
        config.update({
            "base_model": keras.layers.serialize(self.base_model),
        }
        )

    @classmethod
    def from_config(cls, config):
        config["base_model"] = keras.layers.deserialize(
                                                config.pop("base_model"))
        return cls(**config)
