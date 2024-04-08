import keras
import numpy as np
from keras import ops
from teras._src.api_export import teras_export


_VALID_JOIN_METHODS = ['concat', 'add']


@teras_export("teras.layers.TabTransformerColumnEmbedding")
class TabTransformerColumnEmbedding(keras.layers.Layer):
    """
    Column Embedding layer as proposed in the
    "TabTransformer: Tabular Data Modeling Using Contextual Embeddings".

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        cardinalities: list, a list cardinalities of all the features
            in the dataset in the same order as the features' occurrence.
            For numerical features, use any value <=0 as indicator at
            the corresponding index.
            You can use the `compute_cardinalities` function from
            `teras.utils` package for this purpose.
        embedding_dim: int, dimensionality of the embeddings
        use_shared_embedding: bool, whether to use the shared embeddings.
            Defaults to `True`.
            If `False`, this layer will be effectively equivalent to a
            `keras.layers.Embedding` layer.
        shared_embedding_dim: int, dimensionality of the shared embeddings.
            Shared embeddings are the embeddings of the unique column
            identifiers, which according to the paper, help the model
            distinguish categories of one feature from the other.
            By default, its value is set to `embedding_dim / 8` as this
            setup is the most superior in the results presented by the
            authors.
        join_method: str, one of ['concat', 'add'] method to join the
            shared embeddings with feature embeddings.
            Defaults to 'concat', which is the recommended method,
            in which shared embeddings of `shared_embedding_dim` are
            concatenated with `embedding_dim - shared_embedding_dim`
            dimension feature embeddings.
            In 'add', shared embeddings have the same dimensions as the
            features, i.e. the `embedding_dim` and they are element-wise
            added to the features.
    """
    def __init__(self,
                 cardinalities: list,
                 embedding_dim: int,
                 use_shared_embedding: bool = True,
                 shared_embedding_dim: int = None,
                 join_method: str = "concat",
                 **kwargs):
        super().__init__(**kwargs)
        if join_method not in _VALID_JOIN_METHODS:
            raise ValueError(
                f"`join_method` must be one of {_VALID_JOIN_METHODS}, "
                f"but received, {join_method}")
        self.cardinalities = cardinalities
        self.embedding_dim = embedding_dim
        self.use_shared_embedding = use_shared_embedding
        self.shared_embedding_dim = (
            shared_embedding_dim if shared_embedding_dim else
            embedding_dim // 8)
        self.join_method = join_method
        # indices of categorical features
        self._cardinalities_arr = np.array(self.cardinalities)
        self._categorical_idx = np.flatnonzero(
            self._cardinalities_arr != 0)

        categorical_cardinalities = self._cardinalities_arr[self._categorical_idx]

        # The paper assumes each categorical feature has
        # num_categories + 1 embeddings where additional embedding
        # corresponds to a missing value - but we'll just stick to using
        # one extra embedding for the whole embedding table
        num_special_tokens = 1

        # We create a one big lookup table of embeddings for all
        # the categories/classes of categorical features. And we assume
        # the categorical features to be ordinally encoded, hence
        # multiple features will have common values like, [0, 1, 2, ...]
        # So, to distinguish the common values of one feature from the
        # other we create an offset array to offset the values.
        # Additionally, we insert 1 to the leftmost position to reserve
        # the value of 0 for special purpose of 'missing value'.
        categorical_cardinalities = ops.pad(
            categorical_cardinalities,
            (1, 0),
            constant_values=num_special_tokens)
        self._categories_offset = ops.cumsum(
            categorical_cardinalities)[:-1]
        self._categories_offset = ops.cast(self._categories_offset,
                                           "float32")
        self._total_tokens = sum(self.cardinalities) + num_special_tokens

    def build(self, input_shape=None):
        num_categorical_features = len(self._categorical_idx)
        feature_embedding_dim = self.embedding_dim
        shared_embedding_dim = self.shared_embedding_dim

        if self.use_shared_embedding:
            if self.join_method == "concat":
                feature_embedding_dim -= shared_embedding_dim
            # if user specifies any shared embedding dimensions, but the
            # join method is set to add, then shared embeddings will be
            # the same dimensions as the feature embedding and user's
            # supplied value will be overwritten
            else:
                shared_embedding_dim = feature_embedding_dim

        self.feature_embedding = self.add_weight(
            shape=(self._total_tokens, feature_embedding_dim),
            initializer="random_normal",
            trainable=True,
            name="feature_embedding",
        )

        if self.use_shared_embedding:
            self.shared_embedding = self.add_weight(
                shape=(num_categorical_features,
                       shared_embedding_dim),
                initializer="random_normal",
                trainable=True,
                name="shared_embedding"
            )
        self.built = True

    def call(self, inputs):
        _dtype = inputs.dtype
        _batch_size = ops.shape(inputs)[0]
        inputs = ops.take(inputs, indices=self._categorical_idx,
                          axis=1)
        inputs += self._categories_offset
        inputs = ops.cast(inputs, "int32")
        embeddings = ops.take(self.feature_embedding,
                              indices=inputs,
                              axis=0)
        embeddings = ops.cast(embeddings, dtype=_dtype)
        if self.use_shared_embedding:
            if self.join_method == "concat":
                _shared_emb = ops.tile(self.shared_embedding,
                                       (_batch_size, 1, 1))
                embeddings = ops.concatenate([_shared_emb, embeddings],
                                             axis=-1)
            else:
                embeddings += self.shared_embedding
        return embeddings

    def compute_output_shape(self, input_shape):
        return input_shape + (self.embedding_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "cardinalities": self.cardinalities,
            "embedding_dim": self.embedding_dim,
            "use_shared_embedding": self.use_shared_embedding,
            "shared_embedding_dim": self.shared_embedding_dim,
            "join_method": self.join_method
        })
        return config
