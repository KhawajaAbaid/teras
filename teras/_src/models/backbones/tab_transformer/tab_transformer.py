import keras
import numpy as np

from teras._src.layers.categorical_extraction import CategoricalExtraction
from teras._src.layers.continuous_extraction import ContinuousExtraction
from teras._src.layers.tab_transformer.column_embedding import \
    TabTransformerColumnEmbedding
from teras._src.models.backbones.backbone import Backbone
from teras._src.models.backbones.transformer.encoder import \
    TransformerEncoderBackbone
from teras._src.api_export import teras_export


@teras_export("teras.models.TabTransformerBackbone")
class TabTransformerBackbone(Backbone):
    """
    TabTransformer backbone based on the architecture proposed in the
    "TabTransformer: Tabular Data Modeling Using Contextual Embeddings".

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        input_dim: int, dimensionality of the input dataset. i.e. the
            number of features in the dataset.
        cardinalities: list, a list cardinalities of all the features
            in the dataset in the same order as the features' occurrence.
            For numerical features, use any value <=0 as indicator at
            the corresponding index.
            You can use the `compute_cardinalities` function from
            `teras.utils` package for this purpose.
        embedding_dim: int, dimensionality of the embeddings used
            by the model. It is also referred to as the `d_model` or
            model dimensionality.
        use_shared_embedding: bool, whether to use the shared embeddings.
            Defaults to `True`.
            If `False`, this layer will be effectively equivalent to a
            `keras.layers.Embedding` layer.
        shared_embedding_dim: int, dimensionality of the shared embeddings.
            Shared embeddings are the embeddings of the unique column
            identifiers, which according to the paper, help the model
            distinguish categories of one feature from the other.
            By default, its value is set to `embedding_dim / 8` as this
            setup is the most superior in the results presented by the authors.
        join_method: str, one of ['concat', 'add'] method to join the
            shared embeddings with feature embeddings.
            Defaults to 'concat', which is the recommended method,
            in which shared embeddings of `shared_embedding_dim` are
            concatenated with `embedding_dim - shared_embedding_dim`
            dimension feature embeddings.
            In 'add', shared embeddings have the same dimensions as the
            features, i.e. the `embedding_dim` and they are element-wise
            added to the features.
        num_layers: int, number of `TransformerEncoderLayer`s to use in
            the encoder.
        num_heads: int, number of attention heads to use in the
            `MultiHeadAttention` layer.
        feedforward_dim: int, hidden dimensionality to use in the
            `TransformerFeedForward` layer.
        attention_dropout: float, dropout value to use in the
            `MultiHeadAttention` layer. Defaults to 0.
        feedforward_dropout: float, dropout value to use in the
            `TransformerFeedForward` layer. Defaults to 0.
        layer_norm_epsilon: float, epsilon value to use in the
            `LayerNormalization` layer. Defaults to 1e-5.

    Shapes:
        input_shape: (batch_size, input_dim)

        output_shape: (batch_size, input_dim)
    """
    def __init__(self,
                 input_dim: int,
                 cardinalities: list,
                 embedding_dim: int,
                 use_shared_embedding: bool = True,
                 shared_embedding_dim: int = None,
                 join_method: str = "concat",
                 num_layers: int = 6,
                 num_heads: int = 8,
                 feedforward_dim: int = None,
                 attention_dropout: float = 0.,
                 feedforward_dropout: float = 0.,
                 layer_norm_epsilon: float = 1e-5,
                 **kwargs):
        inputs = keras.layers.Input((input_dim,), name="inputs")
        categorical_idx = np.flatnonzero(np.array(cardinalities) != 0)
        continuous_idx = np.flatnonzero(np.array(cardinalities) == 0)

        # Deal with categorical data
        x_cat = CategoricalExtraction(categorical_idx)(inputs)
        x_cat = TabTransformerColumnEmbedding(
            cardinalities=cardinalities,
            embedding_dim=embedding_dim,
            use_shared_embedding=use_shared_embedding,
            shared_embedding_dim=shared_embedding_dim,
            join_method=join_method)(x_cat)
        x_cat = TransformerEncoderBackbone(
            input_dim=len(categorical_idx),
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            attention_dropout=attention_dropout,
            feedforward_dropout=feedforward_dropout,
            layer_norm_epsilon=layer_norm_epsilon)(x_cat)
        x_cat = keras.layers.Flatten()(x_cat)

        # Deal with continuous data
        x_cont = ContinuousExtraction(continuous_idx)(inputs)
        x_cont = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon)(x_cont)

        # Concatenate
        outputs = keras.layers.Concatenate()([x_cat, x_cont])
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.input_dim = input_dim
        self.cardinalities = cardinalities
        self.embedding_dim = embedding_dim
        self.use_shared_embedding = use_shared_embedding
        self.shared_embedding_dim = shared_embedding_dim
        self.join_method = join_method
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def compute_output_shape(self, input_shape):
        cards = np.array(self.cardinalities)
        continuous_dim = sum(cards == 0)
        categorical_dim = sum(cards != 0) * self.embedding_dim
        return input_shape[:1] + (continuous_dim + categorical_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "cardinalities": self.cardinalities,
            "embedding_dim": self.embedding_dim,
            "use_shared_embedding": self.use_shared_embedding,
            "shared_embedding_dim": self.shared_embedding_dim,
            "join_method": self.join_method,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "attention_dropout": self.attention_dropout,
            "feedforward_dropout": self.feedforward_dropout,
            "layer_norm_epsilon": self.layer_norm_epsilon
        })
        return config
