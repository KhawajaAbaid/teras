from tensorflow import keras
from teras.utils.types import ActivationType


@keras.saving.register_keras_serializable(package="teras.layers.vime")
class VimePredictor(keras.layers.Layer):
    """
    Predictor layer based on the architecture proposed by Jinsung Yoon et a.
    in the paper "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        num_labels: `int`, default 32,
            Number of labels to predict

        hidden_dim: ``int``, default 32,
            The hidden dimensionality of the predictor.

        activation: ``str`` or ``callable`` or ``keras.layers.Layer``, default "relu",
            Activation function to use in for the hidden layers.
    """
    def __init__(self,
                 num_labels: int = 2,
                 hidden_dim: int = 32,
                 activation: ActivationType = "relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.predictor_block = keras.models.Sequential(name="predictor_block")
        self.inter_layer_1 = keras.layers.Dense(self.hidden_dim,
                                                activation=self.activation,
                                                name="inter_layer_1")
        self.predictor_block.add(self.inter_layer_1)
        self.inter_layer_2 = keras.layers.Dense(self.hidden_dim,
                                                activation=self.activation,
                                                name="inter_layer_2")
        self.predictor_block.add(self.inter_layer_2)
        self.dense_out = keras.layers.Dense(self.num_labels,
                                            activation=None,
                                            name="dense_out")
        self.predictor_block.add(self.dense_out)
        self.softmax = keras.layers.Softmax()

    def call(self, inputs):
        y_hat_logit = self.predictor_block(inputs)
        y_hat = self.softmax(y_hat_logit)
        return y_hat_logit, y_hat

    def get_config(self):
        config = super().get_config()
        activation_serialized = self.activation
        if not isinstance(self.activation, str):
            activation_serialized = keras.layers.serialize(self.activation)
        config.update({'num_labels': self.num_labels,
                       'hidden_dim': self.hidden_dim,
                       'activation': activation_serialized,
                       }
                      )
        return config
