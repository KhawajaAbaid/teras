import keras

from teras._src.layers.layer_list import LayerList
from teras._src.typing import IntegerSequence, ActivationType
from teras._src.api_export import teras_export


@teras_export("teras.models.GAINDiscriminator")
class GAINDiscriminator(keras.Model):
    """
    Discriminator model for the GAIN architecture proposed by
    Jinsung Yoon et al. in the paper
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Note that the Generator and Discriminator share the exact same
    architecture by default. They differ in the inputs they receive
    and their loss functions.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        data_dim: int, The dimensionality of the input dataset.
            Note the dimensionality must be equal to the dimensionality of
            the transformed dataset that is passed to the fit method and not
            that of original dataset as the dimensionality of raw input dataset
            may change during transformation.
            One way to access the dimensionality of the transformed dataset is
            through the `.data_dim` attribute of the `GAINDataSampler`
            instance used in sampling the dataset.
        hidden_dims: list, A list of hidden dimensionalities for constructing
            hidden block. For each value, a `Dense` layer of that
            dimensionality is added to the hidden block.
            By default, `units_values` = [`data_dim`, `data_dim`].
        activation_hidden: Activation function to use for the hidden layers
            in the hidden block. Defaults to "relu".
        activation_out: Activation function to use for the output layer of
            the Discriminator. Defaults to "sigmoid"
    """
    def __init__(self,
                 data_dim: int,
                 hidden_dims: IntegerSequence = None,
                 activation_hidden: ActivationType = "relu",
                 activation_out: ActivationType = "sigmoid",
                 **kwargs):
        super().__init__(**kwargs)

        if hidden_dims is not None and not isinstance(hidden_dims,
                                                      (list, tuple)):
            raise ValueError(
                "`units_values` must be a list or tuple of units which "
                "determines the number of Discriminator blocks and the "
                f"dimensionality of those blocks. But {hidden_dims} was "
                "passed.")

        self.data_dim = data_dim
        self.hidden_dims = hidden_dims
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out

        if self.hidden_dims is None:
            self.hidden_dims = [self.data_dim] * 2
        self.hidden_block = []
        for dim in self.hidden_dims:
            self.hidden_block.append(
                keras.layers.Dense(
                    units=dim,
                    activation=self.activation_hidden,
                    kernel_initializer="glorot_normal",
                )
            )
        self.hidden_block = LayerList(
            self.hidden_block,
            name="discriminator_hidden_block"
        )

        self.output_layer = keras.layers.Dense(
            self.data_dim,
            activation=self.activation_out,
            name="discriminator_output_layer")

    def build(self, input_shape):
        self.hidden_block.build(input_shape)
        input_shape = self.hidden_block.compute_output_shape(input_shape)
        self.output_layer.build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs is the concatenation of `hint` and manipulated
        # Generator output (i.e. generated samples).
        # `hint` has the same dimensions as data
        # so the inputs received are 2x the dimensions of original data
        x = self.hidden_block(inputs)
        x = self.output_layer(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.data_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'data_dim': self.data_dim,
            'hidden_dims': self.hidden_dims,
            'activation_hidden': self.activation_hidden,
            'activation_out': self.activation_out,
        })
        return config

    @classmethod
    def from_config(cls, config):
        data_dim = config.pop("data_dim")
        return cls(data_dim=data_dim,
                   **config)
