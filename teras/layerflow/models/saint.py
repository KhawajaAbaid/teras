from tensorflow import keras
from tensorflow.keras import backend as K


@keras.saving.register_keras_serializable(package="teras.layerflow.models.backbones")
class SAINT(keras.Model):
    """
    SAINT architecture with LayerFlow design.
    It proposed by Gowthami Somepalli et al.
    in the paper,
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    SAINT performs attention over both rows and columns.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset.

        encoder: ``keras.layers.Layer``,
            An instance of `SAINTEncoder` layer to encode the features embeddings,
            or any layer that can work in its place for that purpose.
            If None, a ``SAINTEncoder`` layer with default values will be used.
            You can import the ``SAINTEncoder`` layer as follows,
                >>> from teras.layerflow.layers import SAINTEncoder

        categorical_feature_embedding: ``keras.layers.Layer``,
            An instance of ``CategoricalFeatureEmbedding`` layer to embedd categorical features
            or any layer that can work in its palce for that purpose.
            If None, a ``CategoricalFeatureEmbedding`` layer with default values will be used.
            You can import the ``CategoricalFeatureEmbedding`` layer as follows,
                >>> from teras.layers import CategoricalFeatureEmbedding

        numerical_feature_embedding: ``keras.layers.Layer``,
            An instance of ``SAINTNumericalFeatureEmbedding`` layer to embedd numerical features
            or any layer that can work in its place for that purpose.
            If None, a ``SAINTNumericalFeatureEmbedding`` layer with default values will be used.
            You can import the ``SAINTNumericalFeatureEmbedding`` layer as follows,
                >>> from teras.layers import SAINTNumericalFeatureEmbedding

        head: ``keras.layers.Layer``,
            An instance of ``Head`` layer to make classification or regression predictions.
            In case you're using this model as a base model for pretraining, you MUST leave
            this argument as None.
    """
    def __init__(self,
                 input_dim: int,
                 encoder: keras.layers.Layer,
                 categorical_feature_embedding: keras.layers.Layer = None,
                 numerical_feature_embedding: keras.layers.Layer = None,
                 head: keras.layers.Layer = None,
                 **kwargs):
        if categorical_feature_embedding is None and numerical_feature_embedding is None:
            raise ValueError("Both `categorical_feature_embedding` and `numerical_feature_embedding` "
                             "cannot be None at the same time as a tabular dataset must contains "
                             "features of at least one of the types if not both. ")

        inputs = keras.layers.Input(shape=(input_dim,))
        x = inputs

        if categorical_feature_embedding is not None:
            x = categorical_feature_embedding(inputs)

        if numerical_feature_embedding is not None:
            numerical_embeddings = numerical_feature_embedding(inputs)

            if categorical_feature_embedding is not None:
                x = K.concatenate([x, numerical_embeddings], axis=1)

        outputs = encoder(x)

        if head is not None:
            outputs = head(x)

        super().__init__(inputs=inputs,
                         outputs=outputs,
                         **kwargs)

        self.input_dim = input_dim
        self.encoder = encoder
        self.categorical_feature_embedding = categorical_feature_embedding
        self.numerical_feature_embedding = numerical_feature_embedding

    def get_config(self):
        config = super().get_config()
        config.update({'input_dim': self.input_dim,
                       'encoder': keras.layers.serialize(self.encoder),
                       'categorical_feature_embedding': keras.layers.serialize(self.categorical_feature_embedding),
                       'numerical_feature_embedding': keras.layers.serialize(self.numerical_feature_embedding),
                       'head': keras.layers.serialize(self.head),
                       })
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        encoder = keras.layers.deserialize(config.pop("encoder"))
        categorical_feature_embedding = keras.layers.deserialize(config.pop("categorical_feature_embedding"))
        numerical_feature_embedding = keras.layers.deserialize(config.pop("numerical_feature_embedding"))
        head = keras.layers.deserialize(config.pop("head"))
        return cls(input_dim,
                   encoder,
                   categorical_feature_embedding=categorical_feature_embedding,
                   numerical_feature_embedding=numerical_feature_embedding,
                   head=head,
                   **config)
