import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers


class NumericalFeatureEmbedding(layers.Layer):
    """
    Numerical Feature Embedding layer as proposed by
    Yury Gorishniy et al. in the paper,
    Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    It simply just projects inputs with  dimensions to Embedding dimensions.
    And the only difference between this NumericalFeatureEmbedding
    and the SAINT's NumericalFeatureEmbedding layer that,
    this layer doesn't use a hidden layer. That's it.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        numerical_features_metadata: `dict`,
            a dictionary of metadata for numerical features
            that maps each feature name to its index in the dataset.
        embedding_dim: `int`, default 32,
            Dimensionality of embeddings, must be the same as the one
            used for embedding categorical features.
    """
    def __init__(self,
                 numerical_features_metadata: dict,
                 embedding_dim: int = 32,
                 ):
        super().__init__()
        self.numerical_features_metadata = numerical_features_metadata
        self.embedding_dim = embedding_dim

        self._num_numerical_features = len(self.numerical_features_metadata)
        # Need to create as many embedding layers as there are numerical features
        self.embedding_layers = []
        for _ in range(self._num_numerical_features):
            self.embedding_layers.append(
                    layers.Dense(units=self.embedding_dim)
                )

        self._is_first_batch = True
        self._is_data_in_dict_format = False

    def call(self, inputs):
        # Find the dataset's format - is it either in dictionary format or array format.
        # If inputs is an instance of dict, it's in dictionary format
        # If inputs is an instance of tuple, it's in array format
        if self._is_first_batch:
            if isinstance(inputs, dict):
                self._is_data_in_dict_format = True
            self._is_first_batch = False

        numerical_feature_embeddings = tf.TensorArray(size=self._num_numerical_features,
                                                      dtype=tf.float32)

        for i, (feature_name, feature_idx) in enumerate(self.numerical_features_metadata.items()):
            if self._is_data_in_dict_format:
                feature = tf.expand_dims(inputs[feature_name], 1)
            else:
                feature = tf.expand_dims(inputs[:, feature_idx], 1)
            embedding = self.embedding_layers[i]
            feature = embedding(feature)
            numerical_feature_embeddings = numerical_feature_embeddings.write(i, feature)

        numerical_feature_embeddings = tf.squeeze(numerical_feature_embeddings.stack())
        if tf.rank(numerical_feature_embeddings) == 3:
            numerical_feature_embeddings = tf.transpose(numerical_feature_embeddings, perm=[1, 0, 2])
        else:
            # else the rank must be 2
            numerical_feature_embeddings = tf.transpose(numerical_feature_embeddings)
        return numerical_feature_embeddings


# TODO rework this layer -- ideally making it simple!
class CLSToken(keras.layers.Layer):
    """
    CLS Token as proposed and implemented by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        embedding_dim: `int`, default 32,
            Embedding dimensions used to embed the numerical
            and categorical features.
        initialization: default "normal",
            Initialization method to use for the weights.
            Must be one of ["uniform", "normal"] if passed as string, otherwise
            you can pass any Keras initializer object.
    """
    def __init__(self,
                 embedding_dim: int = 32,
                 initialization="normal",
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.initialization = initialization

    def build(self, input_shape):
        initializer = self.initialization
        if isinstance(initializer, str):
            assert self.initialization.lower() in ["uniform", "normal"], \
                ("If passed by string, only uniform and normal values are supported. "
                 "Please pass keras.initializers.<YourChoiceOfInitializer> instance"
                 " if you want to use a different initializer.")
            initializer = initializers.random_uniform if self.initialization == "uniform" \
                else initializers.random_normal
        self.weight = self.add_weight(initializer=initializer,
                                      shape=(self.embedding_dim,))

    def expand(self, *leading_dimensions: int) -> tf.Tensor:
        if not leading_dimensions:
            return self.weight
        return tf.broadcast_to(tf.expand_dims(self.weight, axis=0), (*leading_dimensions, self.weight.shape[0]))

    def call(self, inputs):
        return tf.concat([inputs, self.expand(tf.shape(inputs)[0], 1)], axis=1)
