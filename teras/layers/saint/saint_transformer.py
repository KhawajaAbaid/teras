from tensorflow import keras
from teras.layerflow.layers.saint.saint_transformer import SAINTTransformer as _SAINTTransformerLF
from teras.layers.saint.multi_head_inter_sample_attention import MultiHeadInterSampleAttention
from teras.layers.common.transformer import Transformer, FeedForward


@keras.saving.register_keras_serializable(package="teras.layers.saint")
class SAINTTransformer(_SAINTTransformerLF):
    """
    SAINT Transformer layer as proposed by Gowthami Somepalli et al.
    in the paper SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.
    It differs from the usual Transformer (L) block in that it contains additional
    multihead intersample attention layer in addition to the usual multihead attention layer

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        data_dim: ``int``,
            Dimensionality of the input dataset,
            or the total number of features in the dataset.

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
        multihead_inter_sample_attention = MultiHeadInterSampleAttention(
            num_heads=num_inter_sample_attention_heads,
            key_dim=embedding_dim * data_dim,
            dropout=inter_sample_attention_dropout,
            name="inter_sample_multihead_attention"
        )
        feed_forward = FeedForward(embedding_dim=embedding_dim,
                                   multiplier=feedforward_multiplier,
                                   dropout=feedforward_dropout)
        transformer = Transformer(embedding_dim=embedding_dim,
                                  num_attention_heads=num_attention_heads,
                                  attention_dropout=attention_dropout,
                                  feedforward_dropout=feedforward_dropout,
                                  feedforward_multiplier=feedforward_multiplier,
                                  norm_epsilon=norm_epsilon,
                                  name="inner_trasnformer_block_for_features")

        super().__init__(multi_head_inter_sample_attention=multihead_inter_sample_attention,
                         feed_forward=feed_forward,
                         transformer=transformer,
                         **kwargs)

        self.data_dim = data_dim
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
                  'num_embedded_features': self.num_embedded_features,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        # data_dim is the only positional argument
        data_dim = config.pop("data_dim")
        return cls(data_dim=data_dim, **config)
