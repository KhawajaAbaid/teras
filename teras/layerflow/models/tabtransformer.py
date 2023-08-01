import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layerflow.models")
class TabTransformer(keras.Model):
    """
    TabTransformer model class with LayerFlow design.

    TabTransformer architecture is proposed by Xin Huang et al.
    in the paper,
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    TabTransformer is a novel deep tabular data modeling architecture for
    supervised and semi-supervised learning.
    The TabTransformer is built upon self-attention based Transformers.
    The Transformer layers transform the embeddings of categorical features
    into robust contextual embeddings to achieve higher prediction accuracy.

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset.

        categorical_feature_embedding: ``layers.Layer``,
            An instance of ``CategoricalFeatureEmbedding`` layer to embedd categorical features
            or any layer that can work in its place for that purpose.
            If None, a ``CategoricalFeatureEmbedding`` layer with default values will be used.
            You can import the ``CategoricalFeatureEmbedding`` layer as follows,
                >>> from teras.layerflow.layers import CategoricalFeatureEmbedding

        column_embedding: ``layers.Layer``,
            An instance of ``TabTColumnEmbedding`` layer to apply over categorical embeddings,
            or any layer that can work in its place for that purpose.
            If None, a ``TabTColumnEmbedding`` layer with default values will be used.
            You can import the ``TabTColumnEmbedding`` layer as follows,
                >>> from teras.layers import TabTransformerColumnEmbedding

        encoder: ``layers.Layer``,
            An instance of ``Encoder`` layer to encode feature embeddings,
            or any layer that can work in its place for that purpose.
            If None, an ``Encoder`` layer with default values will be used.
            You can import the ``Encoder`` layer as follows,
                >>> from teras.layerflow.layers import Encoder

        numerical_feature_normalization: ``layers.Layer``,
            An instance of ``NumericalFeatureNormalization`` layer to normalize numerical features,
            or any layer that can work in its place for that purpose.
            If None, an ``NumericalFeatureNormalization`` layer with default values will be used.
            You can import the ``NumericalFeatureNormalization`` layer as follows,
                >>> from teras.layerflow.layers import NumericalFeatureNormalization

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
                 column_embedding: keras.layers.Layer = None,
                 encoder: keras.layers.Layer = None,
                 numerical_feature_normalization: keras.layers.Layer = None,
                 head: keras.layers.Layer = None,
                 **kwargs):
        if categorical_feature_embedding is None and numerical_feature_normalization is None:
            raise ValueError("Both `categorical_feature_embedding` and `numerical_feature_normalization` "
                             "cannot be None at the same time as a tabular dataset must contains "
                             "features of at least one of the types if not both. ")

        if isinstance(input_dim, int):
            input_dim = (input_dim,)
        inputs = keras.layers.Input(shape=input_dim,
                                    name="inputs")
        categorical_out = None
        if categorical_feature_embedding is not None:
            x = categorical_feature_embedding(inputs)

            if column_embedding is not None:
                x = column_embedding(x)

            if encoder is None:
                raise ValueError("`encoder` is required to encode the categorical embedding, "
                                 "but received `None`. "
                                 "Please pass an instance of `Encode` layer. "
                                 "You can import it as, `from teras.layerflow.layers import Encoder`")
            x = encoder(x)
            x = keras.layers.Flatten()(x)
            categorical_out = x

        if numerical_feature_normalization is not None:
            numerical_out = numerical_feature_normalization(inputs)
            if categorical_out is not None:
                x = keras.layers.Concatenate()([categorical_out, numerical_out])
            else:
                x = numerical_out

        outputs = x
        if head is not None:
            outputs = head(outputs)

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.input_dim = input_dim
        self.categorical_feature_embedding = categorical_feature_embedding
        self.column_embedding = column_embedding
        self.encoder = encoder
        self.numerical_feature_normalization = numerical_feature_normalization
        self.head = head

    def get_config(self):
        config = super().get_config()
        config.update({'input_dim': self.input_dim,
                       'categorical_feature_embedding': keras.layers.serialize(self.categorical_feature_embedding),
                       'column_embedding': keras.layers.serialize(self.column_embedding),
                       'encoder': keras.layers.serialize(self.encoder),
                       'numerical_feature_normalization': keras.layers.serialize(self.numerical_feature_normalization),
                       'head': keras.layers.serialize(self.head),
                       })
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        categorical_feature_embedding = keras.layers.deserialize(config.pop("categorical_feature_embedding"))
        column_embedding = keras.layers.deserialize(config.pop("column_embedding"))
        encoder = keras.layers.deserialize(config.pop("encoder"))
        numerical_feature_normalization = keras.layers.deserialize(config.pop("numerical_feature_normalization"))
        head = keras.layers.deserialize(config.pop("head"))
        return cls(input_dim=input_dim,
                   categorical_feature_embedding=categorical_feature_embedding,
                   column_embedding=column_embedding,
                   encoder=encoder,
                   numerical_feature_normalization=numerical_feature_normalization,
                   head=head,
                   **config)


@keras.saving.register_keras_serializable(package="teras.layerflow.models")
class TabTransformerPretrainer(keras.Model):
    """
    Pretrainer model for TabTransformer based on the
    Replaced Token Detection (RTD) method.
    RTD replaces the original feature by a random value
    of that feature.
    Here, the loss is minimized for a binary classifier
    that tries to predict whether or not the feature has
    been replaced.

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        model: ``TabTransformer``,
            An instance of base ``TabTransformer`` class to pretrain.

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

        replace_rate: ``float``, default 0.3,
            Fraction of total features per sample to replace.
            Must be in between 0. - 1.0
    """

    def __init__(self,
                 model: TabTransformer,
                 features_metadata: dict,
                 replace_rate: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.replace_rate = replace_rate
        self.features_metadata = features_metadata

        self._categorical_features_exist = len(self.features_metadata["categorical"]) > 0
        self.num_features = len(self.features_metadata["categorical"]) + len(self.features_metadata["numerical"])
        self.num_features_to_replace = tf.cast(tf.cast(self.num_features, tf.float32) * self.replace_rate,
                                               dtype=tf.int32)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.head = keras.layers.Dense(self.num_features,
                                       activation="sigmoid",
                                       name="pretrainer_head")

    def get_pretrained_model(self):
        """Returns pretrained model"""
        return self.model

    @property
    def pretrained_model(self):
        """Returns pretrained model"""
        return self.get_pretrained_model()

    def compile(self,
                loss=keras.losses.BinaryCrossentropy(),
                optimizer=keras.optimizers.AdamW(learning_rate=0.01),
                **kwargs):
        super().compile(**kwargs)
        self.loss_fn = loss
        self.optimizer = optimizer

    def call(self, inputs, mask=None):
        # Since in RTD, for a sample, we randomly replace k% of its features values using
        # random values of those features.
        # We can efficiently achieve this by first getting x_rand = shuffle(inputs)
        # then, to apply replacement, inputs = (inputs * (1-mask)) + (x_rand * mask)
        x_rand = tf.random.shuffle(inputs)
        inputs = (inputs * (1 - mask)) + (x_rand * mask)
        intermediate_features = self.model(inputs)
        # Using sigmoid we essentially get a predicted mask which can we used to
        # compute loss just like that of binary classification
        mask_pred = self.head(intermediate_features)
        return mask_pred

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        # Feature indices is of the shape batch_size x num_features_to_replace
        feature_indices_to_replace = tf.random.uniform((tf.shape(data)[0], self.num_features_to_replace),
                                                       maxval=self.num_features,
                                                       dtype=tf.int32)
        # Mask is of the shape batch_size x num_features and contains values in range [0, 1]
        # A value of 1 represents if the feature is replaced, while 0 means it's not replaced.
        mask = tf.reduce_max(tf.one_hot(feature_indices_to_replace,
                                        depth=self.num_features),
                             axis=1)

        with tf.GradientTape() as tape:
            mask_pred = self(data, mask=mask)
            loss = self.loss_fn(mask, mask_pred)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(mask, mask_pred)
        results = {m.name: m.result() for m in self.metrics}
        return results

    def get_config(self):
        config = super().get_config()
        new_config = {'model': keras.layers.serialize(self.model),
                      'features_metadata': self.features_metadata,
                      'replace_rate': self.replace_rate
                      }
        config.update(new_config)
        return config

    @classmethod
    def from_config(cls, config):
        model = keras.layers.deserialize(config.pop("model"))
        features_metadata = config.pop("features_metadata")
        return cls(model=model,
                   features_metadata=features_metadata,
                   **config)
