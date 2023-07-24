import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layers.node")
class FeatureSelector(keras.layers.Layer):
    """
    FeatureSelector layer for the ``NODE`` architecture.
    This is not part of the official architecture,
    it offers very basic functionality of selecting features
    in case the user specifies the max_features argument in
    the NODE architecture.
    Strictly speaking, it isn't really needed, but since
    in Teras, we're following a functional approach for our
    models and avoiding the pure subclassing where one
    overrides the ``call`` method. Hence, **our** user friendly
    functional approach necessitates the creation of a layer
    that can handle the extra functionality of feature selection
    in the ``call`` method without modifying the other ``NODE``
    architecture-specific layers such as ``ObliviousDecisionTree``

    Args:
        max_features: ``int``, default None,
            Maximum number of features to use.
            If None, all features in the input dataset will be used,
            and the ``FeatureSelector`` layer returns inputs as is.
    """
    def __init__(self,
                 max_features: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_features = max_features

    def call(self, inputs):
        x = inputs
        initial_features = x.shape[-1]
        if self.max_features is None:
            return x
        tail_features = min(self.max_features, x.shape[-1]) - initial_features
        if tail_features != 0:
            x = tf.concat([x[..., :initial_features], x[..., -tail_features:]],
                          axis=-1)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"max_features": self.max_features})
        return config