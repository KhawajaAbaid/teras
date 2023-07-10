from tensorflow.keras import layers


class GeneratorBlock(layers.Layer):
    """
    Residual Block for Generator as used by the authors of CTGAN
    proposed in the paper Modeling Tabular data using Conditional GAN.

    outputs = Concat([ReLU(BatchNorm(Dense(inputs))), inputs])

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        units: Hidden dimensionality of the dense layer
    """
    def __init__(self,
                units: int,
                **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = layers.Dense(self.units)
        self.batch_norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.concat = layers.Concatenate(axis=1)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.batch_norm(x)
        x = self.relu(x)
        out = self.concat([x, inputs])
        return out

    def get_config(self):
        base_config = super().get_config()
        config = {'units': self.units}
        return base_config.update(config)


class DiscriminatorBlock(layers.Layer):
    """
    Discriminator Block based on the architecture proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    outputs = Dropout(LeakyReLU(Dense(inputs)))

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        units: Hidden dimensionality of the dense layer
        leaky_relu_alpha: alpha value to use for leaky relu activation
            Defaults to 0.2.
        dropout_rate: Dropout rate to use in the dropout layer
    """
    def __init__(self,
                 units: int,
                 leaky_relu_alpha: float = 0.2,
                 dropout_rate: float = 0.,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout_rate = dropout_rate

        self.dense = layers.Dense(units)
        self.leaky_relu = layers.LeakyReLU(alpha=self.leaky_relu_alpha)
        self.dropout = layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs):
        out = self.dense(inputs)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        return out

    def get_config(self):
        config = super().get_config()
        new_config = {'units': self.units,
                      'leaky_relu_alpha': self.leaky_relu_alpha,
                      'dropout_rate': self.dropout_rate}
        config.update(new_config)
        return config
