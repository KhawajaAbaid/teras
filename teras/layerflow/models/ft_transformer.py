from tensorflow import keras


@keras.saving.register_keras_serializable("teras.layerflow.models")
class FTTransformer(keras.Model):
    """
    FTTransformer architecture with LayrFlow design.
    FT-Transformer is proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset.

        categorical_feature_embedding: ``keras.layers.Layer``,
            An instance of ``CategoricalFeatureEmbedding`` layer to embedd categorical features
            or any layer that can work in its place.
            You can import the ``CategoricalFeatureEmbedding`` layer as follows,
                >>> from teras.layers import CategoricalFeatureEmbedding

        numerical_feature_embedding: ``keras.layers.Layer``,
            An instance of ``FTNumericalFeatureEmbedding`` layer to embedd numerical features
            or any layer that can work in its place.
            You can import the ``FTNumericalFeatureEmbedding`` layer as follows,
                >>> from teras.layers import FTNumericalFeatureEmbedding

        cls_token: ``keras.layers.Layer``,
            An instance of ``FTCLSToken`` layer, it acts as BeRT-like CLS token,
            or any layer than can work in its place.
            You can import the ``FTCLSToken`` layer as follows,
                >>> from teras.layers import FTCLSToken

        encoder: ``keras.layers.Layer``,
            An instance of ``Encoder`` layer to encode the features embeddings,
            or any layer that can work in its palce.
            You can import the `Encoder` layer as follows,
                >>> from teras.layerflow.layers import Encoder

        head: ``keras.layers.Layer``,
            An instance of either ``ClassificationHead`` or ``RegressionHead`` layers,
            depending on the task at hand.

            REMEMBER: In case you're using this model as a base model for pretraining, you MUST leave
            this argument as None.

            You can import the ``ClassificationHead`` and ``RegressionHead`` layers as follows,
                >>> from teras.layers import ClassificationHead
                >>> from teras.layers import RegressionHead
    """
    def __init__(self,
                 input_dim: int,
                 categorical_feature_embedding: keras.layers.Layer = None,
                 numerical_feature_embedding: keras.layers.Layer = None,
                 cls_token: keras.layers.Layer = None,
                 encoder: keras.layers.Layer = None,
                 head: keras.layers.Layer = None,
                 **kwargs):
        if categorical_feature_embedding is None and numerical_feature_embedding is None:
            raise ValueError("Both `categorical_feature_embedding` and `numerical_feature_embedding` "
                             "cannot be None at the same time as a tabular dataset must contains "
                             "features of at least one of the types if not both. ")
        if cls_token is None:
            raise ValueError("`cls_token` cannot be None. Please pass an instance of `FTCLSToken` layer. "
                             "You can import it as, `from teras.layers import FTCLSToken`")

        if encoder is None:
            raise ValueError("`encoder` cannot be None. Please pass an instance of `Encoder` layer. "
                             "You can import it as, `from teras.layerflow.layers import Encoder``")

        if isinstance(input_dim, int):
            input_dim = (input_dim,)

        inputs = keras.layers.Input(shape=input_dim)
        categorical_out = None

        if categorical_feature_embedding is not None:
            x = categorical_feature_embedding(inputs)
            categorical_out = x

        if numerical_feature_embedding is not None:
            x = numerical_feature_embedding(inputs)
            if categorical_out is not None:
                x = keras.layers.Concatenate(axis=1)([categorical_out, x])
        x = cls_token(x)
        outputs = encoder(x)

        if head is not None:
            outputs = head(outputs)

        super().__init__(inputs=inputs,
                         outputs=outputs,
                         **kwargs)
        self.input_dim = input_dim
        self.categorical_feature_embedding = categorical_feature_embedding
        self.numerical_feature_embedding = numerical_feature_embedding
        self.cls_token = cls_token
        self.encoder = encoder
        self.head = head

    def get_config(self):
        config = super().get_config()
        config.update({'input_dim': self.input_dim,
                       'categorical_feature_embedding': keras.layers.serialize(self.categorical_feature_embedding),
                       'numerical_feature_embedding': keras.layers.serialize(self.numerical_feature_embedding),
                       'cls_token': keras.layers.serialize(self.cls_token),
                       'encoder': keras.layers.serialize(self.encoder),
                       'head': keras.layers.serialize(self.head),
                       })
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        categorical_feature_embedding = keras.layers.deserialize(config.pop("categorical_feature_embedding"))
        numerical_feature_embedding = keras.layers.deserialize(config.pop("numerical_feature_embedding"))
        cls_token = keras.layers.deserialize(config.pop("cls_token"))
        encoder = keras.layers.deserialize(config.pop("encoder"))
        head = keras.layers.deserialize(config.pop("head"))
        return cls(input_dim=input_dim,
                   categorical_feature_embedding=categorical_feature_embedding,
                   numerical_feature_embedding=numerical_feature_embedding,
                   cls_token=cls_token,
                   encoder=encoder,
                   head=head,
                   **config)
