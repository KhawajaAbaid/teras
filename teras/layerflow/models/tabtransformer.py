import keras
from keras import ops, random
from keras.backend import backend
if backend() == "tensorflow":
    import tensorflow as tf
elif backend() == "torch":
    import torch
elif backend() == "jax":
    import jax


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
    into robust contextual embeddings to achieve higher prediction accuracy

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset.

        categorical_feature_embedding: ``layers.Layer``,
            An instance of ``CategoricalFeatureEmbedding`` layer to embedd
            categorical features
            or any layer that can work in its place for that purpose.
            You can import the ``CategoricalFeatureEmbedding`` layer as
            follows,
                >>> from teras.layers import CategoricalFeatureEmbedding

        column_embedding: ``layers.Layer``,
            An instance of ``TabTColumnEmbedding`` layer to apply over
            categorical embeddings,
            or any layer that can work in its place for that purpose.
            You can import the ``TabTColumnEmbedding`` layer as follows,
                >>> from teras.layers import TabTransformerColumnEmbedding

        encoder: ``layers.Layer``,
            An instance of ``Encoder`` layer to encode feature embeddings,
            or any layer that can work in its place for that purpose.
            You can import the ``Encoder`` layer as follows,
                >>> from teras.layerflow.layers import Encoder

        numerical_feature_normalization: ``layers.Layer``,
            An instance of ``NumericalFeatureNormalization`` layer to
            normalize numerical features,
            or any layer that can work in its place for that purpose.
            You can import the ``NumericalFeatureNormalization`` layer as
            follows,
                >>> from teras.layers import NumericalFeatureNormalization

        head: ``keras.layers.Layer``,
            An instance of either ``ClassificationHead`` or
            ``RegressionHead`` layers,
            depending on the task at hand.

            REMEMBER: In case you're using this model as a base model for
            pretraining, you MUST leave
            this argument as None.

            You can import the ``ClassificationHead`` and
            ``RegressionHead`` layers as follows,
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
        if (categorical_feature_embedding is None
                and numerical_feature_normalization is None):
            raise ValueError("Both `categorical_feature_embedding` and "
                             "`numerical_feature_normalization` cannot be "
                             "None at the same time as a tabular dataset "
                             "must contains features of at least one of "
                             "the types if not both. ")

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
                raise ValueError(
                    "`encoder` is required to encode the "
                    "categorical embedding, but received `None`. "
                    "Please pass an instance of `Encode` layer."
                    "You can import it as, "
                    "`from teras.layerflow.layers import Encoder`")
            x = encoder(x)
            x = keras.layers.Flatten()(x)
            categorical_out = x

        if numerical_feature_normalization is not None:
            numerical_out = numerical_feature_normalization(inputs)
            if categorical_out is not None:
                x = keras.layers.Concatenate()(
                    [categorical_out, numerical_out])
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
        self.numerical_feature_normalization = (
            numerical_feature_normalization)
        self.head = head

    def get_config(self):
        config = super().get_config()
        config.update({'input_dim': self.input_dim,
                       'categorical_feature_embedding':
                           keras.layers.serialize(
                               self.categorical_feature_embedding),
                       'column_embedding': keras.layers.serialize(
                           self.column_embedding),
                       'encoder': keras.layers.serialize(self.encoder),
                       'numerical_feature_normalization':
                           keras.layers.serialize(
                               self.numerical_feature_normalization),
                       'head': keras.layers.serialize(self.head),
                       })
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        categorical_feature_embedding = keras.layers.deserialize(
            config.pop("categorical_feature_embedding"))
        column_embedding = keras.layers.deserialize(
            config.pop("column_embedding"))
        encoder = keras.layers.deserialize(config.pop("encoder"))
        numerical_feature_normalization = keras.layers.deserialize(
            config.pop("numerical_feature_normalization"))
        head = keras.layers.deserialize(config.pop("head"))
        return cls(
            input_dim=input_dim,
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
            categorical sub-dictionary is a mapping of categorical feature
            names to a tuple of feature indices and the lists of unique
            values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature
            names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in
            categorical features.
            `{feature_name: feature_idx}` for feature in numerical features
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

        self._categorical_features_exist = len(
            self.features_metadata["categorical"]) > 0
        self.num_features = (len(self.features_metadata["categorical"]) +
                             len(self.features_metadata["numerical"]))
        self.num_features_to_replace = ops.cast(ops.cast(
            self.num_features, keras.backend.floatx()) * self.replace_rate,
                                                dtype="int32")
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
        # Since in RTD, for a sample, we randomly replace k% of its
        # features values using
        # random values of those features.
        # We can efficiently achieve this by first getting
        # x_rand = shuffle(inputs)
        # then, to apply replacement,
        # inputs = (inputs * (1-mask)) + (x_rand * mask)
        x_rand = random.shuffle(inputs)
        inputs = (inputs * (1 - mask)) + (x_rand * mask)
        intermediate_features = self.model(inputs)
        # Using sigmoid we essentially get a predicted mask which can we
        # used to compute loss just like that of binary classification
        mask_pred = self.head(intermediate_features)
        return mask_pred

    def pre_train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        # Feature indices is of the shape
        # batch_size x num_features_to_replace
        feature_indices_to_replace = random.uniform(
            (ops.shape(data)[0],
             self.num_features_to_replace),
            maxval=self.num_features,
            dtype="int32")
        # Mask is of the shape batch_size x num_features and contains
        # values in range [0, 1]
        # A value of 1 represents if the feature is replaced,
        # while 0 means it's not replaced.
        mask = ops.max(ops.one_hot(feature_indices_to_replace,
                                   depth=self.num_features),
                       axis=1)
        return data, mask

    if backend() == "tensorflow":
        def train_step(self, data):
            data, mask = self.pre_train_step(data)
            with tf.GradientTape() as tape:
                mask_pred = self(data, mask=mask)
                loss = self.loss_fn(mask, mask_pred)
            gradients = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply(gradients,
                                 self.trainable_weights)
            self.loss_tracker.update_state(loss)
            # self.compiled_metrics.update_state(mask, mask_pred)
            # for metric in self.metrics:
            #     metric.update_state(mask, mask_pred)
            # TODO: User metrics not support atm, sorry <3
            #   it's 2 AM of 2023 christmas eve and my brain is in another
            #   dimension
            results = {"loss: ", self.loss_tracker.result()}
            return results

    elif backend() == "torch":
        def train_step(self, data):
            data, mask = self.pre_train_step(data)
            data.requires_grad = True
            mask.requires_grad = True
            mask_pred = self(data, mask=mask)
            loss = self.loss_fn(mask, mask_pred)
            loss.backward()
            trainable_weights = [v for v in self.trainable_weights]
            gradients = [v.value.grad for v in trainable_weights]
            with torch.no_grad():
                self.optimizer.apply(gradients,
                                     trainable_weights)
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

    elif backend() == "jax":
        def compute_loss_and_updates(self,
                                     trainable_variables,
                                     non_trainable_variables,
                                     x,
                                     mask,
                                     training=True):
            mask_pred, trainable_variables = self.stateless_call(
                trainable_variables,
                non_trainable_variables,
                x,
                mask,
                training=training
            )
            loss = self.loss_fn(mask, mask_pred)
            return loss, (mask_pred, non_trainable_variables)

        def train_step(self, state, data):
            (trainable_variables,
             non_trainable_variables,
             optimizer_variables,
             metrics_variables
             ) = state
            data, mask = self.pre_train_step(data)
            grad_fn = jax.value_and_grad(self.compute_loss_and_updates,
                                         has_aux=True)
            (loss, (mask_pred, non_trainable_variables)), grads = grad_fn(
                trainable_variables,
                non_trainable_variables,
                data,
                mask,
                training=True
            )

            # Optimizer
            (trainable_variables,
             optimizer_variables) = self.optimizer.stateless_apply(
                optimizer_variables,
                grads,
                trainable_variables
            )

            # Update metrics
            logs = {}
            new_metrics_vars = []
            for metric in self.metrics:
                this_metric_vars = metrics_variables[
                                   len(new_metrics_vars): len(new_metrics_vars) +
                                                          len(metric.varaibles)
                                   ]
                if metric.name == "loss":
                    this_metric_vars = metric.stateless_update_state(
                        this_metric_vars, loss)
                else:
                    this_metric_vars = metric.stateless_update_state(
                        this_metric_vars, mask, mask_pred
                    )
                logs[metric.name] = metric.stateless_result(this_metric_vars)
                new_metrics_vars += this_metric_vars

            state = (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                new_metrics_vars
            )
            return logs, state

    @property
    def metrics(self):
        _metrics = super().metrics()
        _metrics.append(self.loss_tracker)
        return _metrics

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
