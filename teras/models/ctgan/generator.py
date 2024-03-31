import keras
from keras import ops
from teras.layers.ctgan.generator_layer import CTGANGeneratorLayer
from teras.layers.activation import GumbelSoftmax
from teras.utils.types import IntegerSequence
from teras.api_export import teras_export


@teras_export("teras.models.CTGANGenerator")
class CTGANGenerator(keras.Model):
    """
    CTGANGenerator for CTGAN architecture as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        data_dim: int, The dimensionality of the dataset.
            It will also be the dimensionality of the output produced
            by the generator.
            Note the dimensionality must be equal to the dimensionality of
            dataset that is passed to the fit method and not necessarily
            the dimensionality of the raw input dataset as sometimes
            data transformation alters the dimensionality of the dataset.
        metadata: dict, `CTGANGenerator` applies different activation functions
            to its outputs depending on the type of features (categorical or
            numerical). And to determine the feature types and for other
            computations during the activation step, the ``metadata``
            computed during the data transformation step, is required.
            It can be accessed through the `.metadata` property attribute of
            the `CTGANDataTransformer` instance which was used to transform
            the raw input data.
            Note that, this is NOT the same metadata as `features_metadata`,
            which is computed using the `get_metadata_for_embedding` utility
            function from `teras.utils`.
            You must access it through the `.metadata` property attribute of the
            `CTGANDataTransformer`.
        hidden_dims: Sequence, A sequence of integers that is used to
            construct the hidden block.
            For each value, a `CTGANGeneratorLayer` of that dimensionality is
            added. Defaults to [256, 256]
    """
    def __init__(self,
                 data_dim: int,
                 metadata: dict,
                 hidden_dims: IntegerSequence = (256, 256),
                 **kwargs):
        super().__init__(**kwargs)
        self.data_dim = data_dim
        self.metadata = metadata
        self.hidden_dims = hidden_dims

        self.hidden_block = keras.models.Sequential(
            name="generator_hidden_block")
        for dim in self.hidden_dims:
            self.hidden_block.add(CTGANGeneratorLayer(dim))
        self.output_layer = keras.layers.Dense(
            self.data_dim,
            name="generator_output_layer")
        self.gumbel_softmax = GumbelSoftmax()

    def build(self, input_shape):
        self.hidden_block.build(input_shape)
        input_shape = (input_shape[:-1], self.hidden_dims[-1])
        self.output_layer.build(input_shape)

    def apply_activations_by_feature_type(self, interim_outputs):
        """
        This function applies activation functions to the interim outputs
        of the Generator by feature type.
        As CTGAN architecture requires specific transformations on the raw
        input data,
        that decompose one feature in several features,
        and since each type of feature, i.e. numerical or categorical
        require different activation functions to be applied, the process
        of applying those activations becomes rather tricky as it
        requires knowledge of underlying data transformation and
        features metadata.
        To ease the user's burden, in case a user wants to subclass this
        Generator model and completely customize the inner workings of the
        generator but would want to use the activation method specific
        to the CTGAN architecture, so that the subclassed Generator can
        work seamlessly with the rest of the architecture and there
        won't be any discrepancies in outputs produced by the subclasses
        Generator and those expected by the architecture,
        this function is separated, so user can just call this function on
        the interim outputs in the `call` method.

        Args:
            interim_outputs: Outputs produced by the `output_layer` of the
            Generator.

        Returns:
            Final outputs activated by the relevant activation functions.
        """
        outputs = []
        numerical_features_relative_indices = (
            self.metadata)["numerical"]["relative_indices_all"]
        features_relative_indices_all = (
            self.metadata)["relative_indices_all"]
        num_valid_clusters_all = (
            self.metadata)["numerical"]["num_valid_clusters_all"]
        cont_i = 0
        cat_i = 0
        num_categories_all = (
            self.metadata)["categorical"]["num_categories_all"]
        for i, index in enumerate(features_relative_indices_all):
            # the first k = num_numerical_features are numerical in the
            # data
            if i < len(numerical_features_relative_indices):
                # each numerical features has been transformed into
                # num_valid_clusters + 1 features
                # where the first feature is alpha while the following
                # features are beta components
                alphas = ops.tanh(interim_outputs[:, index])
                alphas = ops.expand_dims(alphas, 1)
                outputs.append(alphas)
                betas = self.gumbel_softmax(
                    interim_outputs[:, index + 1: index + 1 + num_valid_clusters_all[cont_i]])
                outputs.append(betas)
                cont_i += 1
            # elif index in categorical_features_relative_indices:
            else:
                # each categorical feature has been converted into a
                # one hot vector
                # of size num_categories
                ds = self.gumbel_softmax(
                    interim_outputs[:, index: index + num_categories_all[cat_i]])
                outputs.append(ds)
                cat_i += 1
        outputs = ops.concatenate(outputs, axis=1)
        return outputs

    def call(self, inputs):
        # inputs have the shape |z| + |cond|
        # while the outputs will have the shape of equal to
        # (batch_size, transformed_data_dims)
        interim_outputs = self.output_layer(self.hidden_block(inputs))
        outputs = self.apply_activations_by_feature_type(interim_outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.data_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'data_dim': self.data_dim,
            'metadata': self.metadata,
            'hidden_dims': self.hidden_dims,
        }
        )
        return config

    @classmethod
    def from_config(cls, config):
        data_dim = config.pop("data_dim")
        return cls(data_dim=data_dim,
                   **config)
