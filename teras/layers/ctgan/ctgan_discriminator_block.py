from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layers.ctgan")
class CTGANDiscriminatorBlock(keras.layers.Layer):
    """
    Discriminator Block based on the architecture proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    outputs = Dropout(LeakyReLU(Dense(inputs)))

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        units: ``int``, default 256,
            Dimensionality of the hidden layer

        leaky_relu_alpha: ``float``, default 0.2,
            Alpha value to use for leaky relu activation

        dropout_rate: ``float``, default 0.,
            Dropout rate to use in the ``Dropout`` layer,
            which is applied after hidden layer.
    """
    def __init__(self,
                 units: int = 256,
                 leaky_relu_alpha: float = 0.2,
                 dropout_rate: float = 0.,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout_rate = dropout_rate

        self.dense = keras.layers.Dense(units)
        self.leaky_relu = keras.layers.LeakyReLU(alpha=self.leaky_relu_alpha)
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs):
        out = self.dense(inputs)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units,
                       'leaky_relu_alpha': self.leaky_relu_alpha,
                       'dropout_rate': self.dropout_rate}
                      )
        return config

