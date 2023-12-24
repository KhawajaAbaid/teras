import keras
from keras import ops, random
from teras.layers.numerical_features_extractor import NumericalFeaturesExtractor
from teras.losses.tabnet import (reconstruction_loss as
                                 reconstruction_loss_fn)
from keras.backend import backend
if backend() == "tensorflow":
    import tensorflow as tf
elif backend() == "torch":
    import torch
elif backend() == "jax":
    import jax

# TODO: Add a default callback that resets the reconstruction loss tracker
#       metric? Maybe we won't need it anymore if out workaround of
#       customizing the metrics property method works... :)

@keras.saving.register_keras_serializable(package="keras.layerflow.models")
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
        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the input dataset.

        features_metadata: ``dict``,
            A nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            ``{feature_name: (feature_idx, vocabulary)}`` for feature in categorical features.
            ``{feature_name: feature_idx}`` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        categorical_feature_embedding: ``keras.layers.Layer``,
            An instance of ``CategoricalFeatureEmbedding`` layer to embedd categorical features
            or any layer that can work in its place.
            You can import the ``CategoricalFeatureEmbedding`` layer as follows,
                >>> from teras.layers import CategoricalFeatureEmbedding

            Important Note: The embedding layer must have a dimensionality of ``1`` else it will
            result in error.


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
                 input_dim: int,
                 features_metadata: dict,
                 categorical_feature_embedding: keras.layers.Layer = None,
                 encoder: keras.layers.Layer = None,
                 head: keras.layers.Layer = None,
                 **kwargs):
        numerical_features_exist = len(features_metadata["numerical"]) > 0
        categorical_features_exist = len(features_metadata["categorical"]) > 0

        if categorical_feature_embedding is None and categorical_features_exist:
            raise ValueError("`categorical_feature_embedding` cannot be None when categorical features exist "
                             "in the dataset. Please pass an instance of `CategoricalFeatureEmbedding` which you "
                             "can import as follows, \n"
                             "`from teras.layers import CategoricalFeatureEmbedding`")

        inputs = keras.layers.Input(shape=(input_dim,))
        if encoder is None:
            raise ValueError("`encoder` cannot be None. Please pass an instance of `TabNetEncoder` layer. "
                             "You can import it as, `from teras.layers import TabNetEncoder``")

        # Important to note that, this is one model where both numerical and categorical feature embeddings
        # can be None. Because tabnet, by default uses raw numerical features.

        if categorical_feature_embedding is None:
            # then there's only numerical features
            features_embeddings = NumericalFeaturesExtractor(features_metadata)(inputs)
        else:
            # then there are definitely categorical features but might or might not be numerical features
            features_embeddings = categorical_feature_embedding(inputs)
            features_embeddings = keras.layers.Flatten()(features_embeddings)
            if numerical_features_exist:
                numerical_embeddings = NumericalFeaturesExtractor(features_metadata)(inputs)
                features_embeddings = keras.layers.Concatenate(axis=1)([features_embeddings, numerical_embeddings])
        outputs = encoder(features_embeddings)
        if head is not None:
            outputs = head(outputs)
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         **kwargs)
        self.input_dim = input_dim
        self.features_metadata = features_metadata
        self.categorical_feature_embedding = categorical_feature_embedding
        self.encoder = encoder
        self.head = head

    def get_config(self):
        config = super().get_config()
        config.update({'input_dim': self.input_dim,
                       'features_metadata': self.features_metadata,
                       'categorical_feature_embedding': keras.layers.serialize(self.categorical_feature_embedding),
                       'encoder': keras.layers.serialize(self.encoder),
                       'head': keras.layers.serialize(self.head),
                       }
                      )
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        features_metadata = config.pop("features_metadata")
        categorical_feature_embedding = keras.layers.deserialize(config.pop("categorical_feature_embedding"))
        encoder = keras.layers.deserialize(config.pop("encoder"))
        head = keras.layers.deserialize(config.pop("head"))
        return cls(input_dim=input_dim,
                   features_metadata=features_metadata,
                   categorical_feature_embedding=categorical_feature_embedding,
                   encoder=encoder,
                   head=head,
                   **config)


@keras.saving.register_keras_serializable(package="keras.layerflow.models")
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

        features_metadata: ``dict``,
            A nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            ``{feature_name: (feature_idx, vocabulary)}`` for feature in categorical features.
            ``{feature_name: feature_idx}`` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        decoder: ``keras.layers.Layer``,
            An instance of ``TabNetDecoder`` layer or any custom layer
            that can be used in its place to reconstruct the input
            features from the encoded representations.
            You can import the ``TabNetDecoder`` layer as
                >>> from teras.layers import TabNetDecoder

        missing_feature_probability: ``float``, default 0.3,
            Fraction of features to randomly mask i.e. make them missing.
            Missing features are introduced in the pretraining dataset and
            the probability of missing features is controlled by the parameter.
            The pretraining objective is to predict values for these missing features,
            (pre)training the ``TabNet`` model in the process.
    """
    def __init__(self,
                 model: TabNet,
                 features_metadata: dict,
                 decoder: keras.layers.Layer = None,
                 missing_feature_probability: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.features_metadata = features_metadata
        self.decoder = decoder
        self.missing_feature_probability = missing_feature_probability

        self.numerical_features_indices = list(self.features_metadata["numerical"].values())
        self._categorical_features_exist = len(self.features_metadata["categorical"]) > 0
        self._numerical_features_exist = len(self.features_metadata["numerical"]) > 0
        self.input_dim = len(self.features_metadata["categorical"]) + len(self.features_metadata["numerical"])

        # self.binary_mask_generator = tfp.distributions.Binomial(total_count=1,
        #                                                         probs=self.missing_feature_probability,
        #                                                         name="binary_mask_generator")
        self._reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

    def get_pretrained_model(self):
        """Returns pretrained model"""
        return self.model

    @property
    def pretrained_model(self):
        """Returns pretrained model"""
        return self.model

    @property
    def feature_importances_per_sample(self):
        """Returns feature importances per sample computed during training."""
        return ops.concatenate(self.encoder.feature_importances_per_sample, axis=0)

    @property
    def feature_importances(self):
        """Returns average feature importances across samples computed during training."""
        return ops.mean(self.feature_importances_per_sample, axis=0)

    def compile(self,
                reconstruction_loss=reconstruction_loss_fn,
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                **kwargs):
        super().compile(optimizer=optimizer,
                        **kwargs)
        self.reconstruction_loss = reconstruction_loss
        # self.optimizer = optimizer

    def call(self, inputs, mask=None):
        # this mask below is what `S` means in the paper, where if an index contains
        # value 1, it means that it is missing
        encoder_input = (1 - mask) * inputs
        # The paper says,
        # The TabNet encoder inputs (1 − S) · f, where f is the original features
        # and the decoder outputs the reconstructed features, S · f^, where f^ is the reconstructed features
        # We initialize P[0] = (1 − S) in encoder so that the model emphasizes merely on the known features.
        # -- So we pass the mask from here, the encoder checks if it received a value for mask, if so it won't
        # initialized the `mask_values` variable in its call method to zeros.
        encoded_representations = self.model.encoder(encoder_input, mask=(1 - mask))
        decoder_outputs = self.decoder(encoded_representations)
        return decoder_outputs

    def pre_train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        cat_embed_dim = 0
        num_embed_dim = 0
        embedded_inputs = None
        if self._categorical_features_exist:
            categorical_embeddings = (
                self.model.categorical_feature_embedding(data))
            # flatten it
            categorical_embeddings = ops.ravel(categorical_embeddings)
            cat_embed_dim = ops.shape(categorical_embeddings)[1]
            embedded_inputs = categorical_embeddings
        # We need to concatenate numerical features to the categorical embeddings
        if self._numerical_features_exist:
            if (hasattr(self.model, "numerical_feature_embedding")
                    and self.model.numerical_feature_embedding is not None):
                numerical_embeddings = (
                    self.model.numerical_feature_embedding(data))
            else:
                numerical_embeddings = ops.take(
                    data,
                    indices=self.numerical_features_indices,
                    axis=1)
            num_embed_dim = ops.shape(numerical_embeddings)[1]
            if embedded_inputs is None:
                embedded_inputs = numerical_embeddings
            else:
                embedded_inputs = ops.concatenate(
                    [embedded_inputs, numerical_embeddings], axis=1)
        total_dim = cat_embed_dim + num_embed_dim
        # Generate mask to create missing samples
        batch_size = ops.shape(embedded_inputs)[0]
        mask = random.binomial(
            shape=(batch_size, total_dim),
            counts=1,
            probabilities=self.missing_feature_probability)
        return embedded_inputs, mask, total_dim

    if backend() == "tensorflow":
        def train_step(self, data):
            embedded_inputs, mask, total_dim = self.pre_train_step(data)
            embedded_inputs.set_shape((None, total_dim))
            with tf.GradientTape() as tape:
                tape.watch(embedded_inputs)
                tape.watch(mask)
                # Reconstruct samples
                reconstructed_samples = self(embedded_inputs, mask=mask)
                # Compute reconstruction loss
                loss = self.loss(
                    real_samples=embedded_inputs,
                    reconstructed_samples=reconstructed_samples,
                    mask=mask)
            gradients = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply(
                grads=gradients,
                trainable_variables=self.trainable_weights)
            self._reconstruction_loss_tracker.update_state(loss)

            for metric in self.metrics:
                metric.update_state(embedded_inputs, reconstructed_samples)

            if self._compile_loss is not None:
                self.compute_loss(x=None,
                                  y=embedded_inputs,
                                  y_pred=reconstructed_samples)
            results = {m.name: m.result() for m in self.metrics}
            # results["reconstruction_loss"] = (
            #                  self._reconstruction_loss_tracker.result())
            return results

    elif backend() == "torch":
        def train_step(self, data):
            embedded_inputs, mask, total_dim = self.pre_train_step(data)
            embedded_inputs.requires_grad = True
            mask.requires_grad = True
            # Reconstruct samples
            reconstructed_samples = self(embedded_inputs, mask=mask)
            # Compute reconstruction loss
            loss = self.loss(
                real_samples=embedded_inputs,
                reconstructed_samples=reconstructed_samples,
                mask=mask)
            loss.backward()

            trainable_weights = [v for v in self.trainable_weights]
            gradients = [v.value.grad for v in trainable_weights]

            with torch.no_grad():
                self.optimizer.apply(
                    grads=gradients,
                    trainable_variables=self.trainable_weights)
            self._reconstruction_loss_tracker.update_state(loss)

            for metric in self.metrics:
                metric.update_state(embedded_inputs, reconstructed_samples)

            if self._compile_loss is not None:
                self.compute_loss(x=None,
                                  y=embedded_inputs,
                                  y_pred=reconstructed_samples)
            results = {m.name: m.result() for m in self.metrics}
            return results

    elif backend() == "jax":
        def compute_loss_and_updates(
                self,
                trainable_variables,
                non_trainable_variables,
                x,
                mask,
                training=False
        ):
            reconstructed_samples, non_trainable_variables = (
                self.stateless_call(
                    trainable_variables,
                    non_trainable_variables,
                    x,
                    mask,
                    training=training
                )
            )
            # TODO: currently only computing reconstruction loss
            #       no checking for self._compiled_loss etc.
            reconstruction_loss = self.reconstruction_loss(
                real_samples=x,
                reconstructed_samples=reconstructed_samples,
                mask=mask
            )
            return reconstruction_loss, (reconstructed_samples,
                                         non_trainable_variables)

        def train_step(self, state, data):
            (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                metrics_variables,
            ) = state
            embedded_inputs, mask, total_dim = self.pre_train_step(data)
            grad_fn = jax.value_and_grad(self.compute_loss_and_updates,
                                         has_aux=True)
            ((loss, (reconstructed_samples, non_trainable_variables)),
             grads) = (
                grad_fn(
                    trainable_variables,
                    non_trainable_variables,
                    embedded_inputs,
                    mask,
                    training=True
                )
            )

            # Update trainable variables and optimizer variables
            trainable_variables, optimizer_variables = (
                self.optimizer.stateless_apply(
                    optimizer_variables,
                    grads,
                    trainable_variables
                )
            )

            # Update metrics
            logs = {}
            new_metrics_vars = []
            for metric in self.metrics:
                this_metric_vars = metrics_variables[
                                   len(new_metrics_vars): len(new_metrics_vars) + len(metric.variables)
                                   ]
                if (metric.name == "reconstruction_loss" or
                        metric.name == "loss"):
                    this_metric_vars = metric.stateless_update_state(
                        this_metric_vars,
                        loss
                    )
                else:
                    this_metric_vars = metric.stateless_update_state(
                        this_metric_vars,
                        embedded_inputs,
                        reconstructed_samples
                    )
                logs[metric.name] = metric.stateless_result(
                    this_metric_vars
                )
                new_metrics_vars += this_metric_vars
            # Return metric logs and updated state variables.
            state = (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                new_metrics_vars,
            )
            return logs, state


    @property
    def metrics(self):
        _metrics = super().metrics()
        _metrics.append(self._reconstruction_loss_tracker)
        return _metrics

    def get_config(self):
        config = super().get_config()
        config.update({'model': keras.layers.serialize(self.model),
                       'features_metadata': self.features_metadata,
                       'decoder': keras.layers.serialize(self.decoder),
                       'missing_feature_probability':
                           self.missing_feature_probability
                       }
                      )
        return config

    @classmethod
    def from_config(cls, config):
        model = keras.layers.deserialize(config.pop("model"))
        features_metadata = config.pop("features_metadata")
        decoder = keras.layers.deserialize(config.pop("decoder"))
        return cls(model=model,
                   features_metadata=features_metadata,
                   decoder=decoder,
                   **config)
