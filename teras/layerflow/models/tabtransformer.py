import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import losses, optimizers
from teras.layers.tabtransformer import (ColumnEmbedding,
                                         ClassificationHead,
                                         RegressionHead)
from teras.layers.embedding import CategoricalFeatureEmbedding
from teras.layers.common.transformer import Encoder
from teras.layerflow.layers.normalization import NumericalFeatureNormalization


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
        features_metadata: `dict`,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                                                                        numerical_features,
                                                                        categorical_features)
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
                >>> from teras.layerflow.layers import TabTColumnEmbedding

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

        head: ``layers.Layer``,
            An instance of ``TabTClassificationHead`` or ``TabTRegressionHead`` layer for final outputs,
            or any layer that can work in place of a head layer for that purpose.
    """
    def __init__(self,
                 features_metadata: dict,
                 categorical_feature_embedding: layers.Layer = None,
                 column_embedding: layers.Layer = None,
                 encoder: layers.Layer = None,
                 numerical_feature_normalization: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        num_categorical_features = len(features_metadata["categorical"])
        num_numerical_features = len(features_metadata["numerical"])
        categorical_features_exist = num_categorical_features > 0
        numerical_features_exist = num_numerical_features > 0
        num_features = num_categorical_features + num_numerical_features
        embedding_dim = 1

        inputs = keras.layers.Input(shape=(num_features,),
                                    name="inputs")
        if categorical_features_exist:
            if categorical_feature_embedding is None:
                categorical_feature_embedding = CategoricalFeatureEmbedding(features_metadata=features_metadata,
                                                                            name="categorical_feature_embedding")
            x = categorical_feature_embedding(inputs)
            embedding_dim = categorical_feature_embedding.embedding_dim

            if column_embedding is None:
                column_embedding = ColumnEmbedding(num_categorical_features=num_categorical_features,
                                                       name="tabtransformer_column_embedding")
            x = column_embedding(x)

            if encoder is None:
                encoder = Encoder(name="encoder")
            x = encoder(x)
            x = layers.Flatten()(x)

        if numerical_features_exist:
            if numerical_feature_normalization is None:
                numerical_feature_normalization = NumericalFeatureNormalization(features_metadata=features_metadata,
                                                                                name="numerical_feature_normalization")
            numerical_out = numerical_feature_normalization(inputs)
            if categorical_features_exist:
                x = layers.Concatenate()([x, numerical_out])
            else:
                x = numerical_out
        new_dimensions = num_categorical_features * embedding_dim + num_numerical_features
        x.set_shape((None, new_dimensions))
        if head is not None:
            x = head(x)

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.features_metadata = features_metadata
        self.categorical_feature_embedding = categorical_feature_embedding
        self.column_embedding = column_embedding
        self.encoder = encoder
        self.numerical_feature_normalization = numerical_feature_normalization
        self.head = head

    def get_config(self):
        config = super().get_config()
        new_config = {'features_metadata': self.features_metadata,
                      'categorical_feature_embedding': keras.layers.serialize(self.categorical_feature_embedding),
                      'column_embedding': keras.layers.serialize(self.column_embedding),
                      'encoder': keras.layers.serialize(self.encoder),
                      'numerical_feature_normalization': keras.layers.serialize(self.numerical_feature_normalization),
                      'head': keras.layers.serialize(self.head),
                      }
        config.update(new_config)
        return config

    @classmethod
    def from_config(cls, config):
        features_metadata = config.pop("features_metadata")
        categorical_feature_embedding = keras.layers.deserialize(config.pop("categorical_feature_embedding"))
        column_embedding = keras.layers.deserialize(config.pop("column_embedding"))
        encoder = keras.layers.deserialize(config.pop("encoder"))
        numerical_feature_normalization = keras.layers.deserialize(config.pop("numerical_feature_normalization"))
        head = keras.layers.deserialize(config.pop("head"))
        return cls(features_metadata=features_metadata,
                   categorical_feature_embedding=categorical_feature_embedding,
                   column_embedding=column_embedding,
                   encoder=encoder,
                   numerical_feature_normalization=numerical_feature_normalization,
                   head=head,
                   **config)


class TabTransformerClassifier(TabTransformer):
    """
    TabTransformerClassifier model class with LayerFlow design.

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
        features_metadata: `dict`,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      numerical_features,
                ..                                                      categorical_features)

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
                >>> from teras.layerflow.layers import TabTColumnEmbedding

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

        head: ``layers.Layer``,
            An instance of ``TabTClassificationHead`` layer for the final outputs,
            or any layer that can work in its place for that purpose.
            If None, ``TabTClassificationHead`` layer with default values will be used.
            You can import the ``TabTClassificationHead`` as follows,
                >>> from teras.layerflow.layers import TabTClassificationHead
    """

    def __init__(self,
                 features_metadata: dict,
                 categorical_feature_embedding: layers.Layer = None,
                 column_embedding: layers.Layer = None,
                 encoder: layers.Layer = None,
                 numerical_feature_normalization: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            head = ClassificationHead()
        super().__init__(features_metadata=features_metadata,
                         categorical_feature_embedding=categorical_feature_embedding,
                         column_embedding=column_embedding,
                         encoder=encoder,
                         numerical_feature_normalization=numerical_feature_normalization,
                         head=head,
                         **kwargs)

    @classmethod
    def from_pretrained(cls,
                        pretrained_model: TabTransformer,
                        head: layers.Layer = None
                        ):
        """
        Class method to create a ``TabTransformerClassifier`` model instance from
        a pretrained base ``TabTransformer`` model instance.

        Args:
            pretrained_model: ``TabTransformer``,
                A pretrained base ``TabTransformer`` model instance.
           head: ``layers.Layer``,
                An instance of ``TabTClassificationHead`` layer for the final outputs,
                or any layer that can work in its place for that purpose.
                If None, ``TabTClassificationHead`` layer with default values will be used.
                You can import ``TabTClassificationHead`` as follows,
                    >>> from teras.layerflow.layers import TabTClassificationHead

        Returns:
            A ``TabTransformerClassifier`` instance based of the pretrained model.
        """
        if head is None:
            head = ClassificationHead(name="tabtransformer_classification_head")
        model = keras.models.Sequential([pretrained_model, head],
                                        name="tabtransformer_classifier_pretrained")
        return model


class TabTransformerRegressor(TabTransformer):
    """
    TabTransformerRegressor model class with LayerFlow design.

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
        features_metadata: `dict`,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      numerical_features,
                ..                                                      categorical_features)
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
                >>> from teras.layerflow.layers import TabTColumnEmbedding

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

        head: ``layers.Layer``,
            An instance of ``TabTRegressionHead`` layer for the final outputs,
            or any layer that can work in place of a ``TabTRegressionHead`` layer for that purpose.
            If None, ``TabTRegressionHead`` layer with default values will be used.
            You can import ``TabTRegressionHead`` as follows,
                >>> from teras.layerflow.layers import TabTRegressionHead
    """
    def __init__(self,
                 features_metadata: dict,
                 categorical_feature_embedding: layers.Layer = None,
                 column_embedding: layers.Layer = None,
                 encoder: layers.Layer = None,
                 numerical_feature_normalization: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            num_outputs = 1
            if "num_outputs" in kwargs:
                num_outputs = kwargs.pop("num_outputs")
            head = RegressionHead(num_outputs=num_outputs,
                                  name="tatransformer_regression_head")
        super().__init__(features_metadata=features_metadata,
                         categorical_feature_embedding=categorical_feature_embedding,
                         column_embedding=column_embedding,
                         encoder=encoder,
                         numerical_feature_normalization=numerical_feature_normalization,
                         head=head,
                         **kwargs)

    @classmethod
    def from_pretrained(cls,
                        pretrained_model: TabTransformer,
                        head: layers.Layer = None,
                        ):
        """
        Class method to create a ``TabTransformerRegressor`` model instance from
        a pretrained base ``TabTransformer`` model instance.

        Args:
            pretrained_model: ``TabTransformer``,
                A pretrained base ``TabTransformer`` model instance.
            head: ``layers.Layer``,
                An instance of ``TabTRegressionHead`` layer for the final outputs,
                or any layer that can work in its place for that purpose.
                If None, ``TabTRegressionHead`` layer with default values will be used.
                You can import ``TabTRegressionHead`` as follows,
                    >>> from teras.layerflow.layers import TabTRegressionHead

        Returns:
            A ``TabTransformerRegressor`` instance based of the pretrained model.
        """
        if head is None:
            head = RegressionHead(name="tabtransformer_regression_head")
        model = keras.models.Sequential([pretrained_model, head],
                                        name="tabtransformer_regressor_pretrained")
        return model


@keras.saving.register_keras_serializable(package="keras.layerflow.models")
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
        replace_rate: ``float``, default 0.3,
            Fraction of total features per sample to replace.
            Must be in between 0. - 1.0
    """

    def __init__(self,
                 model: TabTransformer,
                 replace_rate: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.replace_rate = replace_rate
        self.features_metadata = self.model.features_metadata

        self._categorical_features_exist = len(self.features_metadata["categorical"]) > 0
        self._is_first_batch = True
        self.num_features = None
        self.num_features_to_replace = None
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def build(self, input_shape):
        self.head = layers.Dense(input_shape[1],
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
                loss=losses.BinaryCrossentropy(),
                optimizer=optimizers.AdamW(learning_rate=0.01),
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

        if self._is_first_batch:
            self.num_features = tf.shape(data)[1]
            self.num_features_to_replace = tf.cast(tf.cast(self.num_features, tf.float32) * self.replace_rate,
                                                   dtype=tf.int32)
            self._is_first_batch = False

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
                      'replace_rate': self.replace_rate
                      }
        config.update(new_config)
        return config

    @classmethod
    def from_config(cls, config):
        model = keras.layers.deserialize(config.pop("model"))
        return cls(model=model, **config)