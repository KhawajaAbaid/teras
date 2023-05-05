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
        dim: Dimensionality of Mask Estimator layer
        activation: Activation function to use. Defaults to sigmoid.
    """
    def __init__(self,
                 dim,
                 activation="sigmoid",
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.activation = activation
        self.estimator = layers.Dense(self.dim,
                                      activation=self.activation)

    def call(self, inputs):
        return self.estimator(inputs)


class FeatureEstimator(layers.Layer):
    """
    Feature Estimator layer based on the architecture proposed by Jinsung Yoon et a.
    in the paper "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        dim: Dimensionality of Feature Estimator layer
        activation: Activation function to use. Defaults to sigmoid.
    """
    def __init__(self,
                 dim,
                 activation="sigmoid",
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.activation = activation
        self.estimator = layers.Dense(dim,
                                            activation=self.activation)
    def call(self, inputs):
        return self.estimator(inputs)


class Encoder(layers.Layer):
    """
    Encoder based on the architecture proposed by Jinsung Yoon et a.
    in the paper "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        dim: Dimensionality of Encoder layer
        activation: Activation function to use. Defaults to relu.
    """
    def __init__(self,
                 dim,
                 activation="relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.activation = activation
        self.encoder = layers.Dense(dim,
                                    activation=self.activation)

    def call(self, inputs):
        return self.encoder(inputs)

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({'dim': self.dim})
        return config


# Layers for Semi-Supervised part of VIME

class Predictor(layers.Layer):
    """The predictor"""
    def __init__(self,
                 hidden_dim=None,
                 input_dim=None,
                 n_labels=None,
                 activation="relu",
                 batch_size=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_labels = n_labels
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

        self.dense_out = layers.Dense(self.n_labels,
                                            activation=None,
                                            name="dense_out")
        self.predictor_block.add(self.dense_out)

        self.softmax = layers.Softmax()

    def call(self, inputs):
        y_hat_logit = self.predictor_block(inputs)
        y_hat = self.softmax(y_hat_logit)
        return y_hat_logit, y_hat


class MaskGenerationAndCorruption(layers.Layer):
    """
    A stateless layer that generates masks and corrupted samples
    By implementing this functionality in a layer's call method, we can use this layer in a sequential or functional
    manner and this way the layer and hence the generator functions it calls will have access to the batch size dimension
    because we'll precede this layer with an input layer that specifies the batch dimension and input shape in general
    and pack these both in a sequential model or plug them together using functional API
    """
    def __init__(self,
                 p_m=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.p_m = p_m
    def call(self, inputs, **kwargs):
        """inputs: Batch of unlabeled X"""
        print("Shape of inputs: ", tf.shape(inputs))
        print("Got called with p_m: ", self.p_m)
        # Generate Mask Vector
        masked_batch = vime_mask_generator(self.p_m,
                                 inputs)

        # Generate corrupted samples
        _, corrupted_batch = vime_pretext_generator(masked_batch,
                                                inputs)
        return corrupted_batch
