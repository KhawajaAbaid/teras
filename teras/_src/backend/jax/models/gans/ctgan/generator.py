import jax
from keras import ops
from teras._src.backend.common.models.gans.ctgan.generator import BaseCTGANGenerator
from teras._src.typing import IntegerSequence


class CTGANGenerator(BaseCTGANGenerator):
    def __init__(self,
                 data_dim: int,
                 metadata: dict,
                 hidden_dims: IntegerSequence = (256, 256),
                 seed: int = 1337,
                 **kwargs):
        super().__init__(data_dim=data_dim,
                         metadata=metadata,
                         hidden_dims=hidden_dims,
                         seed=seed,
                         **kwargs)

    def apply_activations_by_feature_type(self, interim_outputs):
        """
        This function applies activation functions to the interim outputs
        of the Generator by feature type.
        As CTGAN architecture requires specific transformations on the raw
        input data,
        that decompose one feature in several features,
        and since each type of feature, i.e. continuous or categorical
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
        continuous_features_relative_indices = (
            self.metadata)["continuous"]["relative_indices_all"]
        features_relative_indices_all = (
            self.metadata)["relative_indices_all"]
        num_valid_clusters_all = (
            self.metadata)["continuous"]["num_valid_clusters_all"]
        cont_i = 0
        cat_i = 0
        num_categories_all = (
            self.metadata)["categorical"]["num_categories_all"]
        for i, index in enumerate(features_relative_indices_all):
            # the first k = num_continuous_features are continuous in the
            # data
            if i < len(continuous_features_relative_indices):
                # each continuous features has been transformed into
                # num_valid_clusters + 1 features
                # where the first feature is alpha while the following
                # features are beta components
                alphas = ops.tanh(interim_outputs[:, index])
                alphas = ops.expand_dims(alphas, 1)
                outputs.append(alphas)
                start_idx = index + 1
                slice_size = num_valid_clusters_all[cont_i]
                betas = self.gumbel_softmax(
                    jax.lax.dynamic_slice_in_dim(interim_outputs,
                                                 start_index=start_idx,
                                                 slice_size=slice_size,
                                                 axis=1),
                )
                outputs.append(betas)
                cont_i += 1
            # elif index in categorical_features_relative_indices:
            else:
                # each categorical feature has been converted into a
                # one hot vector
                # of size num_categories
                start_idx = index
                slice_size = num_categories_all[cat_i]
                ds = self.gumbel_softmax(
                    jax.lax.dynamic_slice_in_dim(interim_outputs,
                                                 start_index=start_idx,
                                                 slice_size=slice_size,
                                                 axis=1))
                outputs.append(ds)
                cat_i += 1
        outputs = ops.concatenate(outputs, axis=1)
        return outputs
