from tensorflow import keras
from teras.layers.common.head import ClassificationHead, RegressionHead


class TabNet(keras.Model):
    """
    TabNet model class with LayerFlow design.

    TabNet is a novel high-performance and interpretable canonical
    deep tabular data learning architecture.
    TabNet uses sequential attention to choose which features to reason
    from at each decision step, enabling interpretability and more
    efficient learning as the learning capacity is used for the most
    salient features.

    TabNet is proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        features_metadata: ``dict``,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        categorical_feature_embedding: ``keras.layers.Layer``,
            An instance of `CategoricalFeatureEmbedding` layer to embedd categorical features
            or any layer that can work in place of `CategoricalFeatureEmbedding` for that purpose.
            If None, a `CategoricalFeatureEmbedding` layer with default values will be used.
            You can import the `CategoricalFeatureEmbedding` layer as follows,
                >>> from teras.layers import CategoricalFeatureEmbedding

        encoder: ``keras.layers.Layer``,
            An instance of Encoder layer to encode feature embeddings,
            or any layer that can work in place of Encoder for that purpose.
            If None, an Encoder layer with default values will be used.
                >>> from teras.layers import TabNetEncoder

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
                 features_metadata: dict,
                 categorical_features_embedding: keras.layers.Layer = None,
                 encoder: keras.layers.Layer = None,
                 head: keras.layers.Layer = None,
                 **kwargs):
        super().__init__(features_metadata=features_metadata,
                         **kwargs)
        if categorical_features_embedding is not None:
            self.categorical_features_embedding = categorical_features_embedding

        if encoder is not None:
            self.encoder = encoder

        if head is not None:
            self.head = head

    def get_config(self):
        config = super().get_config()
        new_config = {'categorical_features_embedding': keras.layers.serialize(self.categorical_features_embedding),
                      'encoder': keras.layers.serialize(self.encoder),
                      'head': keras.layers.serialize(self.head),
                      }
        config.update(new_config)
        return config


class TabNetPretrainer(keras.Model):
    """
    TabNetPretrainer with LayerFlow desing.

    It is an encoder-decoder model based on the TabNet architecture,
    where the TabNet model acts as an encoder while a separate decoder
    is used to reconstruct the input features.

    It is proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        model: ``TabNet``,
            An instance of base ``TabNet`` model to pretrain.

        decoder: `keras.layers.Layer`,
            An instance of ``TabNetDecoder`` layer or any custom layer
            that can be used in its place to reconstruct the input
            features from the encoded representations.
            You can import the ``TabNetDecoder`` layer as
                >>> from teras.layers import TabNetDecoder

        missing_feature_probability: ``float``, default 3,
            Fraction of features to randomly mask i.e. make them missing.
            Missing features are introduced in the pretraining dataset and
            the probability of missing features is controlled by the parameter.
            The pretraining objective is to predict values for these missing features,
            (pre)training the ``TabNet`` model in the process.
    """
    def __init__(self,
                 model: TabNet,
                 decoder: keras.layers.Layer = None,
                 missing_feature_probability: float = 0.3,
                 **kwargs):
        super().__init__(model=model,
                         missing_feature_probability=missing_feature_probability,
                         **kwargs)

        if decoder is not None:
            self.decoder = decoder

    def get_config(self):
        config = super().get_config()
        new_config = {'model': keras.layers.serialize(self.model),
                      'decoder': keras.layers.serialize(self.decoder)
                      }
        config.update(new_config)
        return config
