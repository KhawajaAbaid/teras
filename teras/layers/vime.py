import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from teras.utils import vime_mask_generator, vime_pretext_generator


# Layers for Self Supervised part of VIME


class MaskEstimator(layers.Layer):
    """
    Mask Estimator layer based on the architecture proposed by Jinsung Yoon et a.
    in the paper "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        units: `int`, default 32,
            Dimensionality of Mask Estimator layer
        activation: default "sigmoid",
            Activation function to use.
    """
    def __init__(self,
                 units: int = 32,
                 activation="sigmoid",
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.estimator = layers.Dense(self.units,
                                      activation=self.activation)

    def call(self, inputs):
        return self.estimator(inputs)

    def get_config(self):
        config = super().get_config()
        new_config = {'units': self.units,
                      'activation': self.activation,
                      }
        config.update(new_config)
        return config


class FeatureEstimator(layers.Layer):
    """
    Feature Estimator layer based on the architecture proposed by Jinsung Yoon et a.
    in the paper "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        units: `int`, default 32,
            Dimensionality of Feature Estimator layer
        activation: default "sigmoid",
            Activation function to use.
    """
    def __init__(self,
                 units: int = 32,
                 activation="sigmoid",
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.estimator = layers.Dense(self.units,
                                      activation=self.activation)

    def call(self, inputs):
        return self.estimator(inputs)

    def get_config(self):
        config = super().get_config()
        new_config = {'units': self.units,
                      'activation': self.activation,
                      }
        config.update(new_config)
        return config


class Encoder(layers.Layer):
    """
    Encoder based on the architecture proposed by Jinsung Yoon et a.
    in the paper "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        units: `int Dimensionality of Encoder layer
        activation: Activation function to use. Defaults to relu.
    """
    def __init__(self,
                 units: int = 32,
                 activation="relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.encoder = layers.Dense(self.units,
                                    activation=self.activation)

    def call(self, inputs):
        return self.encoder(inputs)

    def get_config(self):
        config = super().get_config()
        new_config = {'units': self.units,
                      'activation': self.activation,
                      }
        config.update(new_config)
        return config


# Layers for Semi-Supervised part of VIME

class Predictor(layers.Layer):
    """
    Predictor layer based on the architecture proposed by Jinsung Yoon et a.
    in the paper "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    # TODO make docs great again!

    Args:
        units: `int`, default 32,
            The hidden dimensionality of the predictor.
        input_dim: `int`,
            Input dimensionality of the dataset
        num_labels: `int`, default 32,
            Number of labels to predict
        activation: default "relu",
            Activation function to use in for the hidden layers.
        batch_size: `int`, default 512,
            Batch size being used.
    """
    def __init__(self,
                 input_dim: int,
                 units: int = 32,
                 num_labels: int = 2,
                 activation="relu",
                 batch_size: int = 512,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.units = units
        self.num_labels = num_labels
        self.activation = activation
        self.batch_size = batch_size

        self.predictor_block = models.Sequential()

        self.input_layer = layers.Input(shape=(None, input_dim,),
                                        batch_size=self.batch_size)
        self.predictor_block.add(self.input_layer)

        self.inter_layer_1 = layers.Dense(self.hidden_dim,
                                          activation=self.activation,
                                          name="inter_layer_1")
        self.predictor_block.add(self.inter_layer_1)

        self.inter_layer_2 = layers.Dense(self.hidden_dim,
                                          activation=self.activation,
                                          name="inter_layer_2")
        self.predictor_block.add(self.inter_layer_2)

        self.dense_out = layers.Dense(self.num_labels,
                                      activation=None,
                                      name="dense_out")
        self.predictor_block.add(self.dense_out)

        self.softmax = layers.Softmax()

    def call(self, inputs):
        y_hat_logit = self.predictor_block(inputs)
        y_hat = self.softmax(y_hat_logit)
        return y_hat_logit, y_hat

    def get_config(self):
        config = super().get_config()
        new_config = {'input_dim': self.input_dim,
                      'units': self.units,
                      'num_labels': self.num_labels,
                      'activation': self.activation,
                      'batch_size': self.batch_size,
                      }
        config.update(new_config)
        return config


class MaskGenerationAndCorruption(layers.Layer):
    """
    A stateless layer that generates masks and corrupted samples
    By implementing this functionality in a layer's call method, we can use this layer in a sequential or functional
    manner and this way the layer and hence the generator functions it calls will have access to the batch size dimension
    because we'll precede this layer with an input layer that specifies the batch dimension and input shape in general
    and pack these both in a sequential model or plug them together using functional API

    Args:
        p_m: `float`, default 0.3,
            Corruption probability. (Note: Don't do it in real life!)
    """
    def __init__(self,
                 p_m: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.p_m = p_m

    def call(self, inputs):
        """inputs: Batch of unlabeled X"""

        # Generate Mask Vector
        mask = tf.random.stateless_binomial(shape=tf.shape(inputs),
                                            seed=(0, 0),
                                            counts=1,
                                            probs=self.p_m,
                                            output_dtype=tf.float32)

        # Generate corrupted samples
        num_samples = tf.shape(inputs)[0]
        dim = tf.shape(inputs)[1]

        X_bar = tf.TensorArray(size=dim, dtype=tf.float32)
        for i in range(dim):
            idx = tf.random.shuffle(tf.range(num_samples))
            X_bar = X_bar.write(i, tf.gather(inputs[:, i], idx))
        X_bar = tf.transpose(X_bar.stack())

        # Corrupt Samples
        X_tilde = inputs * (1 - mask) + X_bar * mask
        return X_tilde

    def get_config(self):
        config = super().get_config()
        new_config = {'p_m': self.input_dim,
                      }
        config.update(new_config)
        return config
