# Base task class
import keras


class Task(keras.Model):
    """
    Base class for building Task models.

    Taken from keras-cv, with some truncation.
    https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/task.py
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backbone = None

    @property
    def backbone(self):
        return self._backbone

    @backbone.setter
    def backbone(self, value):
        self._backbone = value

    def get_config(self):
        return {"name": self.name,
                "trainable": self.trainable}

    @classmethod
    def from_config(cls, config):
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        return cls(**config)
