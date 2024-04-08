import keras

from teras._src.layers.layer_list import LayerList
from teras._src.typing import IntegerSequence
from teras._src.api_export import teras_export


@teras_export("teras.models.PCGAINClassifier")
class PCGAINClassifier(keras.Model):
    """
    The auxiliary classifier for the PC-GAIN architecture proposed by Yufeng
    Wang et al. in the paper
    "PC-GAIN: Pseudo-label Conditional Generative Adversaria Imputation
    Networks for Incomplete Data"

    Reference(s):
        https://arxiv.org/abs/2011.07770

    Args:
        num_classes: int, Number of classes to predict. It should be equal to
            the `num_clusters`, computed during the pseudo label generation.
        data_dim: int, The dimensionality of the input dataset.
            Note the dimensionality must be equal to the dimensionality of
            dataset that is passed to the fit method and not necessarily the
            dimensionality of the raw input dataset as sometimes data
            transformation alters the dimensionality of the dataset.
        hidden_dims: list, A list/tuple of units to construct hidden block of
            classifier. For each element, a new hidden layer will be added.
            By default, `hidden_dims` = [`data_dim`, `data_dim`]
        activation_hidden: Activation function to use for the hidden layers
            in the classifier. Defaults to "relu".
        activation_out: Activation function to use for the output layer of
            classifier. Defaults to "softmax".
    """
    def __init__(self,
                 num_classes: int,
                 data_dim: int,
                 hidden_dims: IntegerSequence = None,
                 activation_hidden: str = "relu",
                 activation_out: str = "softmax",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.data_dim = data_dim
        self.hidden_dims = hidden_dims
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out

        if self.hidden_dims is None:
            self.hidden_dims = [data_dim] * 2

        self.hidden_block = []
        for i, dim in enumerate(self.hidden_dims):
            self.hidden_block.append(
                keras.layers.Dense(
                    dim,
                    activation=self.activation_hidden,
                    name=f"pcgain_classifier_hidden_layer_{i}")
            )
        self.hidden_block = LayerList(self.hidden_block,
                                      sequential=True,
                                      name="pcgain_classifier_hidden_block")
        self.output_layer = keras.layers.Dense(
            self.num_classes,
            activation=self.activation_out,
            name="pcgain_classifier_output_layer")

    def build(self, input_shape):
        self.hidden_block.build(input_shape)
        input_shape = self.hidden_block.compute_output_shape(input_shape)
        self.output_layer.build(input_shape)

    def call(self, inputs):
        x = self.hidden_block(inputs)
        return self.output_layer(x)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.num_classes,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'data_dim': self.data_dim,
            'hidden_dims': self.hidden_dims,
            'activation_hidden': self.activation_hidden,
            'activation_out': self.activation_out
        })
        return config
