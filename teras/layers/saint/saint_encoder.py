from tensorflow import keras
from teras.layerflow.layers.saint.saint_encoder import SAINTEncoder as _SAINTEncoderLF
from teras.layers.saint.saint_transformer import SAINTTransformer


@keras.saving.register_keras_serializable(package="teras.layers.saint")
class SAINTEncoder(_SAINTEncoderLF):
    """
    SAINTEncoder for the SAINT architecture as proposed by
    Gowthami Somepalli et al. in the paper
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.
    It simply stacks N transformer layers and applies them to the outputs
    of the embedded features.

    It differs from the typical Encoder block only in that the Transformer
    layer is a bit different from the regular Transformer layer used in the
    Transformer based architectures as it uses multi-head inter-sample attention,
    in addition to the regular mutli-head attention for features.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        data_dim: ``int``,
            Dimensionality of the input dataset,
            or the total number of features in the dataset.

        num_transformer_layer: ``int``, default 6,
            Number of transformer layers to use in the Encoder

        embedding_dim: ``int``, default 32,
            Embedding dimensions used to embedd numerical and
            categorical features. These server as the key dimensions
            in the MultiHeadAttention layer.

        num_attention_heads: ``int``, default 8,
            Number of heads to use in the typical ``MultiHeadAttention``
            layer that will be applied over features.

        num_inter_sample_attention_heads: ``int``, default 8,
            Number of heads to use in the ``MultiHeadInterSampleAttention``
            that will be applied over rows

        embedding_dim: ``int``, default 32,
            Embedding dimensions. These will also serve as key dimensions
             for the attention layers

        attention_dropout: ``float``, default 0.1,
            Dropout rate for ``MultiHeadAttention`` which is applied over
             features.

        inter_sample_attention_dropout: ``float``, default 0.1,
            Dropout rate for ``MultiHeadInterSampleAttention`` which is
            applied over rows.

        feedforward_dropout: ``float``, default 0.1,
            Dropout rate for the ``Dropout`` layer that is part of the
            ``FeedForward`` layer.

        feedforward_multiplier: ``int``, default 4.
            Multiplier that is multiplied with the ``embedding_dim``
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the ``FeedForward`` layer.

        apply_attention_to_features: ``bool``, default True,
            Whether to apply attention over features.
            If True, the regular ``MultiHeadAttention`` layer will be applied
            over features.

        apply_attention_to_rows: ``bool``, default True,
            Whether to apply attention over rows.
            If True, the ``MultiHeadInterSampleAttention`` will apply attention
            over rows.
            NOTE: It is strongly recommended to keep both as True, but you
            can turn one off for experiment's sake.
            Also, note that, both CANNOT be False at the same time!
    """
    def __init__(self,
                 data_dim: int,
                 num_transformer_layers: int = 6,
                 embedding_dim: int = 32,
                 num_attention_heads: int = 8,
                 num_inter_sample_attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 inter_sample_attention_dropout: float = 0.1,
                 feedforward_dropout: float = 0.1,
                 feedforward_multiplier: int = 4,
                 norm_epsilon: float = 1e-6,
                 apply_attention_to_features: bool = True,
                 apply_attention_to_rows: bool = True,
                 **kwargs):
        if not apply_attention_to_features and not apply_attention_to_rows:
            raise ValueError("`apply_attention_to_features` and `apply_attention_to_rows` both cannot be False "
                             "at the same time. You must set at least one to True if not both. "
                             f"Received: `apply_attention_to_features`={apply_attention_to_features}, "
                             f"`apply_attention_to_rows`={apply_attention_to_rows}")

        saint_transformer_layers = keras.models.Sequential(name="saint_transformer_layers")
        for i in range(num_transformer_layers):
            saint_transformer_layers.add(SAINTTransformer(
                data_dim=data_dim,
                embedding_dim=embedding_dim,
                num_attention_heads=num_attention_heads,
                num_inter_sample_attention_heads=num_inter_sample_attention_heads,
                attention_dropout=attention_dropout,
                inter_sample_attention_dropout=inter_sample_attention_dropout,
                feedforward_dropout=feedforward_dropout,
                feedforward_multiplier=feedforward_multiplier,
                apply_attention_to_features=apply_attention_to_features,
                apply_attention_to_rows=apply_attention_to_rows,
                name=f"saint_transformer_layer_{i}"))

        super().__init__(saint_transformer_layers=saint_transformer_layers,
                         **kwargs)

        self.data_dim = data_dim
        self.num_transformer_layers = num_transformer_layers
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.num_inter_sample_attention_heads = num_inter_sample_attention_heads
        self.attention_dropout = attention_dropout
        self.inter_sample_attention_dropout = inter_sample_attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.feedforward_multiplier = feedforward_multiplier
        self.norm_epsilon = norm_epsilon
        self.apply_attention_to_features = apply_attention_to_features
        self.apply_attention_to_rows = apply_attention_to_rows

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'data_dim': self.data_dim,
                  'num_transformer_layers': self.num_transformer_layers,
                  'embedding_dim': self.embedding_dim,
                  'num_attention_heads': self.num_attention_heads,
                  'num_inter_sample_attention_heads': self.num_inter_sample_attention_heads,
                  'attention_dropout': self.attention_dropout,
                  'inter_sample_attention_dropout': self.inter_sample_attention_dropout,
                  'feedforward_dropout': self.feedforward_dropout,
                  'feedforward_multiplier': self.feedforward_multiplier,
                  'norm_epsilon': self.norm_epsilon,
                  'apply_attention_to_features': self.apply_attention_to_features,
                  'apply_attention_to_rows': self.apply_attention_to_rows,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        # data_dim is the only positional argument
        data_dim = config.pop("data_dim")
        return cls(data_dim=data_dim, **config)
