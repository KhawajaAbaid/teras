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
        data_dim: ``int``,
            The dimensionality of the input dataset,
            or the number of features in the dataset.

        num_labels: `int`, default 32,
            Number of labels to predict

        units: ``int``, default 32,
            The hidden dimensionality of the predictor.

        activation: ``str`` or ``callable`` or ``keras.layers.Layer``, default "relu",
            Activation function to use in for the hidden layers.

        batch_size: ``int``, default 512,
            Batch size being used.
    """
    def __init__(self,
                 data_dim: int = None,
                 num_labels: int = 2,
                 units: int = 32,
                 activation: ActivationType = "relu",
                 batch_size: int = 512,
                 **kwargs):
        super().__init__(**kwargs)
        self.data_dim = data_dim
        self.num_labels = num_labels
        self.units = units
        self.activation = activation
        self.batch_size = batch_size

        self.predictor_block = keras.models.Sequential(name="predictor_block")
        self.input_layer = keras.layers.Input(shape=(None, data_dim,),
                                              batch_size=self.batch_size)
        self.predictor_block.add(self.input_layer)
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
        config.update({'data_dim': self.data_dim,
                       'units': self.units,
                       'num_labels': self.num_labels,
                       'activation': activation_serialized,
                       'batch_size': self.batch_size,
                       }
                      )
        return config

    @classmethod
    def from_config(cls, config):
        data_dim = config.pop("data_dim")
        return cls(data_dim=data_dim,
                   **config)
