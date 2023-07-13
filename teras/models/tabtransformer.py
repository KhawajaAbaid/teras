import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import losses, optimizers
from teras.layers.embedding import CategoricalFeatureEmbedding
from teras.layers.tabtransformer import ColumnEmbedding, ClassificationHead, RegressionHead
from teras.layers.common.transformer import Encoder
from typing import List, Union, Tuple
from teras.config.tabtransformer import TabTransformerConfig
from teras.utils import convert_dict_to_array_tensor
from teras.layers.encoding import LabelEncoding


LIST_OR_TUPLE_OF_INT = Union[List[int], Tuple[int]]
LAYER_OR_MODEL = Union[layers.Layer, keras.Model]


class TabTransformer(keras.Model):
    """
    TabTransformer architecture as proposed by Xin Huang et al.
    in the paper,
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    TabTransformer, a novel deep tabular data modeling architecture for
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
        embedding_dim: `int`, default 32, Dimensionality of the learnable
            feature embeddings for categorical features.
        num_transformer_layers: `int`, default 6, Number of transformer layers
            to use in the encoder.
            The encoder is used to contextualize the learned feature embeddings.
        num_attention_heads: `int`, default 8, Number of attention heads to use
            in the MultiHeadSelfAttention layer that is part of the transformer
            layer which in turn is part of the encoder.
        attention_dropout: `float`, default 0.0, Dropout rate to use in the
            MultiHeadSelfAttention layer in the transformer layer.
        feedforward_dropout: `float`, default 0.0, Dropout rate to use for the
            dropout layer in the FeedForward block.
        feedforward_multiplier: `int`, default 4.
            Multiplier that is multiplied with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.
        norm_epsilon: `float`, default 1e-6, A very small number used for normalization
            in the LayerNormalization layer.
        use_column_embedding: `bool`, default True, Whether to use the novel ColumnEmbedding
            layer proposed in the TabTransformer architecture for the categorical features.
            The ColumnEmbedding layer is an alternative to positional encoding that is applied
            in the Transformers in Natural Langauge Processing application settings.
        encode_categorical_values: `bool`, default True, whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
"""
    def __init__(self,
                 features_metadata: dict,
                 embedding_dim: int = TabTransformerConfig.embedding_dim,
                 num_transformer_layers: int = TabTransformerConfig.num_transformer_layers,
                 num_attention_heads: int = TabTransformerConfig.num_attention_heads,
                 attention_dropout: float = TabTransformerConfig.attention_dropout,
                 feedforward_dropout: float = TabTransformerConfig.feedforward_dropout,
                 feedforward_multiplier: int = TabTransformerConfig.feedforward_multiplier,
                 norm_epsilon: float = TabTransformerConfig.norm_epsilon,
                 use_column_embedding: bool = TabTransformerConfig.use_column_embedding,
                 encode_categorical_values: bool = TabTransformerConfig.encode_categorical_values,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        if features_metadata is None:
            raise ValueError(f"""
            `features_metadata` is required for embedding features and hence cannot be None.            
            You can get this features_metadata dictionary by calling
            `teras.utils.get_categorical_features_vocabulary(dataset, categorical_features, numerical_features)`
            Received, `features_metadata`: {features_metadata}
            """)

        self.features_metadata = features_metadata
        self.embedding_dim = embedding_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.feedforward_multiplier = feedforward_multiplier
        self.use_column_embedding = use_column_embedding
        self.norm_epsilon = norm_epsilon
        self.encode_categorical_values = encode_categorical_values

        self.categorical_features_metadata = self.features_metadata["categorical"]
        self.numerical_features_metadata = self.features_metadata["numerical"]
        self.num_categorical_features = len(self.categorical_features_metadata)
        self.num_numerical_features = len(self.numerical_features_metadata)
        self.num_features = self.num_categorical_features + self.num_numerical_features
        self.numerical_features_exist = self.num_numerical_features > 0
        self.categorical_features_exist = self.num_categorical_features > 0

        # _processed_features_dim is the computed dimensionality of the resultant features
        # after all data has been processed.
        # Since categorical features are first embedded and then flattened, so their shape
        # becomes `number of categorical features` * `embedding dimensions`
        # And since we only normalize the numerical features so their shape stays the same
        # Finally we concatenate the flattened embedded categorical features and normalized
        # numerical features giving us a final shape that is computed as below.
        # (Note that this computation easily handles the cases when either numerical features
        # are categorical features don't exist, in which case the respective _num variable
        # will be 0.)
        # We need this dimensions to set the shape of `features` variable in the `call` method.
        self._processed_features_dim = (self.num_categorical_features * self.embedding_dim) \
                                        + self.num_numerical_features

        if self.categorical_features_exist:
            self.categorical_feature_embedding = CategoricalFeatureEmbedding(
                                                    categorical_features_metadata=self.categorical_features_metadata,
                                                    embedding_dim=self.embedding_dim,
                                                    encode=self.encode_categorical_values
                                                )
            self.column_embedding = ColumnEmbedding(embedding_dim=self.embedding_dim,
                                                    num_categorical_features=self.num_categorical_features)
            self.encoder = Encoder(num_transformer_layers=self.num_transformer_layers,
                                   num_heads=self.num_attention_heads,
                                   embedding_dim=self.embedding_dim,
                                   attention_dropout=self.attention_dropout,
                                   feedforward_dropout=self.feedforward_dropout,
                                   feedforward_multiplier=self.feedforward_multiplier,
                                   norm_epsilon=self.norm_epsilon)
            self.flatten = layers.Flatten()
        self.norm = layers.LayerNormalization(epsilon=self.norm_epsilon)
        self.head = None

        self._is_first_batch = True
        self._is_data_in_dict_format = False

    def reset_training_flags(self):
        """
        Resets the `_is_first_batch` and `_is_data_in_dict_format` flags
        to their default values.

        Training flags like `_is_first_batch` and `_is_data_in_dict_format`
        are vital for training and to handle data of different types and
        to compute/apply different things on the first batch.
        In regular training this works fine but when we pretrain this model
        the flags value are set and they stay the same when pretrained base
        model is mixed with the classificaiton or regression head which
        often results in error.
        For instance, the `_is_first_batch` is set to False during pretraining
        so it will stay the same during finetuning and since the
        `_is_data_in_dict_format` flag is only set if `_is_first_batch` is True,
        so that will stay the same as well.
        Consequently, any difference between data format between pretraining
        and fine-tuning will lead to errors since the model won't be able to
        infer the vital info required to make decisions about what
        functionality to apply based on data.
        """
        self._is_first_batch = True
        self._is_data_in_dict_format = False
        self.categorical_feature_embedding._is_first_batch = True
        self.categorical_feature_embedding._is_data_in_dict_format = False

    def call(self, inputs):
        if self._is_first_batch:
            if isinstance(inputs, dict):
                self._is_data_in_dict_format = True
            self._is_first_batch = False
        # TODO: Shouldn't we convert dict format data to array format using convert_dict_to_array_tensor?
        #   Thoughts? How would it affect performance?

        features = None
        if self.categorical_features_exist:
            # The categorical feature embedding layer takes care of handling
            # different input data types and features names nad indices
            categorical_features = self.categorical_feature_embedding(inputs)
            if self.use_column_embedding:
                categorical_features = self.column_embedding(categorical_features)
            # Contextualize the embedded categorical features
            categorical_features = self.encoder(categorical_features)
            # Flatten the contextualized embeddings of the categorical features
            categorical_features = self.flatten(categorical_features)
            features = categorical_features

        if self.numerical_features_exist:
            # TODO make this efficient -- for loop isn't needed for numerical features if data's in array format
            # In TabTransformer we normalize the raw numerical features
            # and concatenate them with flattened contextualized categorical features
            numerical_features = tf.TensorArray(size=self.num_numerical_features,
                                                dtype=tf.float32)
            for i, (feature_name, feature_idx) in enumerate(self.numerical_features_metadata.items()):
                if self._is_data_in_dict_format:
                    feature = tf.expand_dims(inputs[feature_name], axis=1)
                else:
                    feature = tf.expand_dims(inputs[:, feature_idx], axis=1)
                feature = tf.cast(feature, tf.float32)
                numerical_features = numerical_features.write(i, feature)
            numerical_features = tf.transpose(tf.squeeze(numerical_features.stack()))
            if features is not None:
                features = tf.concat([features, numerical_features],
                                     axis=1)
            else:
                features = numerical_features

        # Since `features` shape is pretty ambigious -- so in graphmode it results in an error
        # since it passes shape (None, None) because it infers both dimensions as None,
        # as both are subjected to change depending on the `if` conditions.
        # To combat that, we manually set the shape using self._processed_features_dim
        # Read more about it in the __init__ method
        features.set_shape((None, self._processed_features_dim))
        outputs = features
        if self.head is not None:
            outputs = self.head(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        new_config = {'features_metadata': self.features_metadata,
                      'embedding_dim': self.embedding_dim,
                      'num_transformer_layers': self.num_transformer_layers,
                      'num_attention_heads': self.num_attention_heads,
                      'attention_dropout': self.attention_dropout,
                      'feedforward_dropout': self.feedforward_dropout,
                      'feedforward_multiplier': self.feedforward_multiplier,
                      'norm_epsilon': self.norm_epsilon,
                      'use_column_embedding': self.use_column_embedding,
                      'encode_categorical_values': self.encode_categorical_values,
                      }
        config.update(new_config)
        return config


class TabTransformerClassifier(TabTransformer):
    """
    TabTransformerClassifier based on the TabTransformer architecture
    as proposed by Xin Huang et al. in the paper,
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    TabTransformer, a novel deep tabular data modeling architecture for
    supervised and semi-supervised learning.
    The TabTransformer is built upon self-attention based Transformers.
    The Transformer layers transform the embeddings of categorical features
    into robust contextual embeddings to achieve higher prediction accuracy.

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        num_classes: `int`, default 2, Number of classes to predict.
        head_units_values: `List[int] | Tuple[int]`, default [64, 32],
            Units values to use in the hidden layers in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
        activation_out: Activation to use in the Classification head,
            by default, `sigmoid` is used for binary and `softmax` is used
            for multi-class classification.
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
        embedding_dim: `int`, default 32, Dimensionality of the learnable
            feature embeddings for categorical features.
        num_transformer_layers: `int`, default 6, Number of transformer layers
            to use in the encoder.
            The encoder is used to contextualize the learned feature embeddings.
        num_attention_heads: `int`, default 8, Number of attention heads to use
            in the MultiHeadSelfAttention layer that is part of the transformer
            layer which in turn is part of the encoder.
        attention_dropout: `float`, default 0.0, Dropout rate to use in the
            MultiHeadSelfAttention layer in the transformer layer.
        feedforward_dropout: `float`, default 0.0, Dropout rate to use for the
            dropout layer in the FeedForward block.
        feedforward_multiplier: `int`, default 4.
            Multiplier that is multiplied with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.
        norm_epsilon: `float`, default 1e-6, A very small number used for normalization
            in the LayerNormalization layer.
        use_column_embedding: `bool`, default True, Whether to use the novel ColumnEmbedding
            layer proposed in the TabTransformer architecture for the categorical features.
            The ColumnEmbedding layer is an alternative to positional encoding that is applied
            in the Transformers in Natural Langauge Processing application settings.
        encode_categorical_values: `bool`, default True, whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
    """
    def __init__(self,
                 num_classes: int = 2,
                 head_units_values: LIST_OR_TUPLE_OF_INT = (64, 32),
                 activation_out=None,
                 features_metadata: dict = None,
                 embedding_dim: int = TabTransformerConfig.embedding_dim,
                 num_transformer_layers: int = TabTransformerConfig.num_transformer_layers,
                 num_attention_heads: int = TabTransformerConfig.num_attention_heads,
                 attention_dropout: float = TabTransformerConfig.attention_dropout,
                 feedforward_dropout: float = TabTransformerConfig.feedforward_dropout,
                 feedforward_multiplier: int = TabTransformerConfig.feedforward_multiplier,
                 norm_epsilon: float = TabTransformerConfig.norm_epsilon,
                 use_column_embedding: bool = TabTransformerConfig.use_column_embedding,
                 encode_categorical_values: bool = TabTransformerConfig.encode_categorical_values,
                 **kwargs
                 ):
        super().__init__(features_metadata=features_metadata,
                         embedding_dim=embedding_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         attention_dropout=attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         feedforward_multiplier=feedforward_multiplier,
                         norm_epsilon=norm_epsilon,
                         use_column_embedding=use_column_embedding,
                         encode_categorical_values=encode_categorical_values,
                         **kwargs)

        self.num_classes = num_classes
        self.head_units_values = head_units_values
        self.activation_out = activation_out
        self.head = ClassificationHead(num_classes=self.num_classes,
                                       units_values=self.head_units_values,
                                       activation_out=self.activation_out,
                                       name="tabtransformer_classification_head")

    @classmethod
    def from_pretrained(cls,
                        pretrained_model: TabTransformer,
                        num_classes: int = 2,
                        head_units_values: LIST_OR_TUPLE_OF_INT = (64, 32),
                        activation_out=None):
        """
        Class method to create a TabTransformer Classifier model instance from
        a pretrained base TabTransformer model instance.

        Args:
            pretrained_model: `TabTransformer`,
                A pretrained base TabTransformer model instance.
            num_classes: `int`, 2,
                Number of classes to predict.
            head_units_values: `List[int] | Tuple[int]`, default (64, 32),
                For each value in the sequence,
                a hidden layer of that dimension is added to the ClassificationHead.
            activation_out:
                Activation function to use in the (head) output layer.

        Returns:
            A TabTransformer Classifier instance based of the pretrained model.
        """
        num_classes = 1 if num_classes <= 2 else num_classes
        if activation_out is None:
            activation_out = "sigmoid" if num_classes == 1 else "softmax"
        # inputs = layers.Input(shape=(pretrained_model.num_features,))
        # x = pretrained_model(inputs, training=False)
        # outputs = ClassificationHead(num_classes=num_classes,
        #                              units_values=head_units_values,
        #                              activation_out=activation_out,
        #                              name="tabtransformer_classification_head")(x)
        # model = models.Model(inputs=inputs, outputs=outputs)
        head = ClassificationHead(num_classes=num_classes,
                                  units_values=head_units_values,
                                  activation_out=activation_out,
                                  name="tabtransformer_classification_head")
        from teras.layerflow.models.simple import SimpleModel
        model = SimpleModel(body=pretrained_model,
                            head=head,
                            name="tabtransformer_classifier_pretrained")
        return model

    def get_config(self):
        config = super().get_config()
        new_config = {'num_classes': self.num_classes,
                      'activation_out': self.activation_out
                      }
        config.update(new_config)
        return config


class TabTransformerRegressor(TabTransformer):
    """
    TabTransformerRegressor based on the TabTransformer architecture
    as proposed by Xin Huang et al. in the paper,
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    TabTransformer, a novel deep tabular data modeling architecture for
    supervised and semi-supervised learning.
    The TabTransformer is built upon self-attention based Transformers.
    The Transformer layers transform the embeddings of categorical features
    into robust contextual embeddings to achieve higher prediction accuracy.

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        num_outputs: `int`, default 1, Number of regression outputs to predict.
        head_units_values: `List[int] | Tuple[int]`, default [64, 32],
            Units values to use in the hidden layers in the Regression head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
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
        embedding_dim: `int`, default 32, Dimensionality of the learnable
            feature embeddings for categorical features.
        num_transformer_layers: `int`, default 6, Number of transformer layers
            to use in the encoder.
            The encoder is used to contextualize the learned feature embeddings.
        num_attention_heads: `int`, default 8, Number of attention heads to use
            in the MultiHeadSelfAttention layer that is part of the transformer
            layer which in turn is part of the encoder.
        attention_dropout: `float`, default 0.0, Dropout rate to use in the
            MultiHeadSelfAttention layer in the transformer layer.
        feedforward_dropout: `float`, default 0.0, Dropout rate to use for the
            dropout layer in the FeedForward block.
        feedforward_multiplier: `int`, default 4.
            Multiplier that is multiplied with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.
        norm_epsilon: `float`, default 1e-6, A very small number used for normalization
            in the LayerNormalization layer.
        use_column_embedding: `bool`, default True, Whether to use the novel ColumnEmbedding
            layer proposed in the TabTransformer architecture for the categorical features.
            The ColumnEmbedding layer is an alternative to positional encoding that is applied
            in the Transformers in Natural Langauge Processing application settings.
        encode_categorical_values: `bool`, default True, whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 head_units_values: LIST_OR_TUPLE_OF_INT = (64, 32),
                 features_metadata: dict = None,
                 embedding_dim: int = TabTransformerConfig.embedding_dim,
                 num_transformer_layers: int = TabTransformerConfig.num_transformer_layers,
                 num_attention_heads: int = TabTransformerConfig.num_attention_heads,
                 attention_dropout: float = TabTransformerConfig.attention_dropout,
                 feedforward_dropout: float = TabTransformerConfig.feedforward_dropout,
                 feedforward_multiplier: int = TabTransformerConfig.feedforward_multiplier,
                 norm_epsilon: float = TabTransformerConfig.norm_epsilon,
                 use_column_embedding: bool = TabTransformerConfig.use_column_embedding,
                 encode_categorical_values: bool = TabTransformerConfig.encode_categorical_values,
                 **kwargs
                 ):
        super().__init__(features_metadata=features_metadata,
                         embedding_dim=embedding_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         attention_dropout=attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         feedforward_multiplier=feedforward_multiplier,
                         norm_epsilon=norm_epsilon,
                         use_column_embedding=use_column_embedding,
                         encode_categorical_values=encode_categorical_values,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head_units_values = head_units_values
        self.head = RegressionHead(num_outputs=self.num_outputs,
                                   units_values=self.head_units_values,
                                   name="tabtransformer_regression_head")

    @classmethod
    def from_pretrained(cls,
                        pretrained_model: TabTransformer,
                        num_outputs: int = 1,
                        head_units_values: LIST_OR_TUPLE_OF_INT = (64, 32)
                        ):
        """
        Class method to create a TabTransformer Regressor model instance from
        a pretrained base TabTransformer model instance.

        Args:
            pretrained_model: `TabTransformer`,
                A pretrained base TabTransformer model instance.
            num_outputs: `int`, 1,
                Number of regression outputs to predict.
            head_units_values: `List[int] | Tuple[int]`, default (64, 32),
                For each value in the sequence,
                a hidden layer of that dimension is added to the ClassificationHead.

        Returns:
            A TabTransformer Regressor instance based of the pretrained model.
        """

        # Functional approach
        # inputs = layers.Input(shape=(pretrained_model.num_features,))
        # x = pretrained_model(inputs, training=False)
        # outputs = RegressionHead(num_outputs=num_outputs,
        #                          units_values=head_units_values,
        #                          name="tabtransformer_regression_head")(x)
        # model = models.Model(inputs=inputs, outputs=outputs)

        # The problem with functional approach is that it cannot handle dictionary format data

        # Sequential approach
        # model = keras.models.Sequential([pretrained_model, head])

        # The problem with sequential is that it gives a plethora of warnings
        # because of dictionary data, saying that layers
        # in a sequential model should only have a one input tensor

        # But subclassed model can handle it all with ease and no warnings.
        # here we used the layerflow api of teras.
        head = RegressionHead(num_outputs=num_outputs,
                              units_values=head_units_values,
                              name="tabtransformer_regression_head")
        from teras.layerflow.models.simple import SimpleModel
        model = SimpleModel(body=pretrained_model,
                            head=head,
                            name="tabtransformer_regressor_pretrained")
        return model

    def get_config(self):
        config = super().get_config()
        new_config = {'num_outputs': self.num_outputs,
                      }
        config.update(new_config)
        return config


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
        model: `TabTransformer`,
            An instance of base TabTransformer class to pretrain.
        replace_rate: `float`, default 0.3,
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

        self.label_encoding = LabelEncoding(categorical_features_metadata=self.model.categorical_features_metadata,
                                            concatenate_numerical_features=True)

        self._is_first_batch = True
        self._is_data_in_dict_format = False
        self.num_features = None
        self.num_features_to_replace = None
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def build(self, input_shape):
        self.head = layers.Dense(input_shape[1], activation="sigmoid")

    def get_pretrained_model(self):
        """Returns pretrained model"""
        self.model.reset_training_flags()
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
            if isinstance(data, dict):
                self._is_data_in_dict_format = True
                self.num_features = len(data)
            else:
                self.num_features = tf.shape(data)[1]
            self.num_features_to_replace = int(self.num_features * self.replace_rate)
            # self.num_features_to_replace = tf.cast(self.num_features_to_replace,
            #                                        dtype=tf.int32)
            self._is_first_batch = False

        # To make things simpler and easier, specifically the shuffling process,
        # we convert data to array format if it's in dict format
        # But first we check if the user has set the `encode_categorical_values` flag, in which case
        # we'll label encoding layer whose outputs will be in array format.
        # When instantiating the LabelEncoding layer we set the `concatenate_numerical_features`
        # flag to True so that the layer returns all features and not just categorical features.
        if self.model.categorical_features_exist and self.model.encode_categorical_values:
            data = self.label_encoding(data)
            # Okay this is a bit of hacky way to do things but --
            # Since the base model is already instantiated and the encode flag is set
            # for the categorical feature embedding layer but here we're encoding things
            # in advance so it will result in error since it will try to encode already encoded
            # values considering them as string values.
            # So to avoid that, we'll set the categorical feature embedding layer's encode flag
            # to False
            self.model.categorical_feature_embedding.encode = False

        elif self._is_data_in_dict_format:
            # If the encode flag is not set then it means that data is in homogeneous format
            # in which case we can safely just convert dictionary format data to array format
            data = convert_dict_to_array_tensor(data)

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
        # Since we cant check for the last batch last epoch so
        # reset it at the end of every batch
        self.model.categorical_feature_embedding.encode = True
        return results

    def get_config(self):
        config = super().get_config()
        new_config = {'model': keras.layers.serialize(self.model),
                      'replace_rate': self.replace_rate
                      }
        config.update(new_config)
        return config
