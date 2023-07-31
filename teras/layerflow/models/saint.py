import tensorflow as tf
from tensorflow import keras
from teras.losses.saint import info_nce_loss, denoising_loss


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
                 encoder: keras.layers.Layer = None,
                 head: keras.layers.Layer = None,
                 **kwargs):
        if categorical_feature_embedding is None and numerical_feature_embedding is None:
            raise ValueError("Both `categorical_feature_embedding` and `numerical_feature_embedding` "
                             "cannot be None at the same time as a tabular dataset must contains "
                             "features of at least one of the types if not both. ")

        if encoder is None:
            raise ValueError("`encoder` cannot be None. Please pass an instance of `SAINTEncoder` layer. "
                             "You can import it as, `from teras.layerflow.layers import SAINTEncoder``")

        inputs = keras.layers.Input(shape=(input_dim,))

        if categorical_feature_embedding is None:
            # then there's only numerical features because we already
            # have a check above that both categorical and numerical embeddings
            # cannot be None at the same time
            feature_embeddings = numerical_feature_embedding(inputs)
        else:
            categorical_embeddings = categorical_feature_embedding(inputs)
            numerical_embeddings = numerical_feature_embedding(inputs)
            feature_embeddings = keras.layers.Concatenate(axis=1)([categorical_embeddings, numerical_embeddings])

        outputs = encoder(feature_embeddings)

        if head is not None:
            outputs = head(outputs)

        super().__init__(inputs=inputs,
                         outputs=outputs,
                         **kwargs)

        self.input_dim = input_dim
        self.encoder = encoder
        self.categorical_feature_embedding = categorical_feature_embedding
        self.numerical_feature_embedding = numerical_feature_embedding
        self.head = head

    def get_config(self):
        config = super().get_config()
        config.update({'input_dim': self.input_dim,
                       'categorical_feature_embedding': keras.layers.serialize(self.categorical_feature_embedding),
                       'numerical_feature_embedding': keras.layers.serialize(self.numerical_feature_embedding),
                       'encoder': keras.layers.serialize(self.encoder),
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
        return cls(input_dim=input_dim,
                   categorical_feature_embedding=categorical_feature_embedding,
                   numerical_feature_embedding=numerical_feature_embedding,
                   encoder=encoder,
                   head=head,
                   **config)


@keras.saving.register_keras_serializable(package="teras.layerflow.models")
class SAINTPretrainer(keras.Model):
    """
    SAINTPretrainer model with LayerFlow design.
    It is based on the pretraining architecture of the SAINT model
    proposed by Gowthami Somepalli et al.
    in the paper,
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    SAINT performs attention over both rows and columns.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        model: ``keras.Model``,
            An instance of the ``SAINT`` model that is to be pretrained.

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

        mixup: ``keras.layers.Layer``,
            An instance of ``MixUp`` layer or any custom layer that can work
            in its place.
            You can import the ``MixUp`` layer as follows,
                >>> from teras.layers import MixUp

        cutmix: ``keras.layers.Layer``,
            An instance of ``CutMix`` layer or any custom layer that can work
            in its place.
            You can import the `CutMix` layer as follows,
                >>> from teras.layers import CutMix

        projection_head_1: ``keras.layers.Layer``,
            An instance of ``SAINTProjectionHead`` layer that is used to project embeddings
            of *real* samples to a lower dimensionality before reconstructing the inputs.
            You can import the ``SAINTProjectionHead`` layer as follows,
                >>> from teras.layerflow.layers import SAINTProjectionHead

        projection_head_2: ``keras.layers.Layer``,
            An instance of ``SAINTProjectionHead`` layer that is used to project embeddings
            of *augmented* samples to a lower dimensionality before reconstructing the inputs.
            You can import the ``SAINTProjectionHead`` layer as follows,
                >>> from teras.layerflow.layers import SAINTProjectionHead

        reconstruction_head: ``keras.layers.Layer``,
            An instance of ``SAINTReconstructionHead`` which applies a separate ``ReconstructionHeadBlock``
            to reconstruct the input features.
            You can import the ``SAINTReconstructionHead`` as follows,
                >>> from teras.layerflow.layers import SAINTReconstructionHead

        temperature: ``float``, default 0.7,
            Temperature value used in the computation of the InfoNCE contrastive loss.

        lambda_: ``float``, default 10,
            Controls the weightage of denoising loss in the summation of denoising and
            contrastive loss.
    """
    def __init__(self,
                 model: keras.Model,
                 features_metadata: dict,
                 mixup: keras.layers.Layer = None,
                 cutmix: keras.layers.Layer = None,
                 projection_head_1: keras.layers.Layer = None,
                 projection_head_2: keras.layers.Layer = None,
                 reconstruction_head: keras.layers.Layer = None,
                 temperature: float = 0.7,
                 lambda_: float = 10.,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        if mixup is None:
            raise ValueError("`mixup` cannot be None. Please pass an instance of `MixUp` layer. "
                             "You can import it as, `from teras.layers import MixUp`")

        if cutmix is None:
            raise ValueError("`cutmix` cannot be None. Please pass an instance of `CutMix` layer. "
                             "You can import it as, `from teras.layers import CutMix``")

        if projection_head_1 is None:
            raise ValueError("`projection_head_1` cannot be None. "
                             "Please pass an instance of `SAINTProjectionHead` layer. "
                             "You can import it as, `from teras.layers import SAINTProjectionHead``")

        if projection_head_2 is None:
            raise ValueError("`projection_head_2` cannot be None. "
                             "Please pass an instance of `SAINTProjectionHead` layer. "
                             "You can import it as, `from teras.layers import SAINTProjectionHead`")

        if reconstruction_head is None:
            raise ValueError("`reconstruction_head` cannot be None. "
                             "Please pass an instance of `SAINTReconstructionHead` layer. "
                             "You can import it as, `from teras.layers import SAINTReconstructionHead`")

        self.features_metadata = features_metadata
        self._categorical_features_exist = len(self.features_metadata["categorical"]) > 0
        self._numerical_features_exist = len(self.features_metadata["numerical"]) > 0

        self.mixup = mixup
        self.cutmix = cutmix
        self.projection_head_1 = projection_head_1
        self.projection_head_2 = projection_head_2
        self.reconstruction_head = reconstruction_head
        self.temperature = temperature
        self.lambda_ = lambda_

        self.contrastive_loss_tracker = keras.metrics.Mean(name="contrastive_loss")
        self.denoising_loss_tracker = keras.metrics.Mean(name="denoising_loss")

    def get_pretrained_model(self):
        """Returns pretrained model"""
        return self.model

    @property
    def pretrained_model(self):
        """Returns pretrained model"""
        return self.model

    def compile(self,
                contrastive_loss=info_nce_loss,
                denoising_loss=denoising_loss,
                **kwargs):
        super().compile(**kwargs)
        self.contrastive_loss = contrastive_loss
        self.denoising_loss = denoising_loss

    def call(self, inputs):
        x = inputs

        # Apply cutmix on the raw input space
        x_prime = self.cutmix(x)

        # Embed the raw inputs as well as cutmixed data
        p = None
        p_prime = None
        if self._categorical_features_exist:
            p = self.model.categorical_feature_embedding(x)
            p_prime = self.model.categorical_feature_embedding(x_prime)

        if self._numerical_features_exist:
            numerical_features = self.model.numerical_feature_embedding(x)
            numerical_features_prime = self.model.numerical_feature_embedding(x_prime)
            if p is not None:
                p = tf.concat([p, numerical_features],
                              axis=1)
                p_prime = tf.concat([p_prime, numerical_features_prime],
                                    axis=1)
            else:
                p = numerical_features
                p_prime = numerical_features_prime

        # Apply mixup on the embedding space -- only to the augment data
        p_prime = self.mixup(p_prime)

        # Pass these embeddings through saint encoder
        r = self.model.encoder(p)
        r_prime = self.model.encoder(p_prime)

        # Pass the encoded features through projection heads
        z = self.projection_head_1(r)
        z_prime = self.projection_head_2(r_prime)
        # Normalize
        z = z / tf.norm(z, axis=-1, keepdims=True)
        z_prime = z_prime / tf.norm(z_prime, axis=-1, keepdims=True)
        # Flatten last two dimensions
        z = tf.reshape(z, shape=(tf.shape(z)[0], tf.reduce_prod(tf.shape(z)[1:])))
        z_prime = tf.reshape(z_prime, shape=(tf.shape(z_prime)[0], tf.reduce_prod(tf.shape(z_prime)[1:])))

        # To reconstruct the input data, we pass the encodings of augmented data
        # i.e. the `r_prime` through a reconstruction head
        reconstructed_samples = self.reconstruction_head(r_prime)

        return z, z_prime, reconstructed_samples

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z, z_prime, reconstructed_samples = self(data)
            c_loss = self.contrastive_loss(real_projection_outputs=z,
                                           augmented_projection_outputs=z_prime,
                                           temperature=self.temperature)
            d_loss = self.denoising_loss(real_samples=data,
                                         reconstructed_samples=reconstructed_samples,
                                         categorical_features_metadata=self.features_metadata["categorical"])

            loss = c_loss + self.lambda_ * d_loss
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.contrastive_loss_tracker.update_state(c_loss)
        self.denoising_loss_tracker.update_state(d_loss)
        # If user has passed any additional metrics to compile, we should update their states
        if len(self.compiled_metrics.metrics) > 0:
            self.compiled_metrics.update_state(data, reconstructed_samples)
        # If user has passed any additional losses to compile, we should call them
        if self.compiled_loss._losses is not None:
            self.compiled_loss(data, reconstructed_samples)
        results = {m.name: m.result() for m in self.metrics}
        return results

    def get_config(self):
        config = super().get_config()
        new_config = {'model': keras.layers.serialize(self.model),
                      'features_metadata': self.features_metadata,
                      'mixup': keras.layers.serialize(self.mixup),
                      'cutmix': keras.layers.serialize(self.cutmix),
                      'projection_head_1': keras.layers.serialize(self.projection_head_1),
                      'projection_head_2': keras.layers.serialize(self.projection_head_2),
                      'reconstruction_head': keras.layers.serialize(self.reconstruction_head),
                      'temperature': self.temperature,
                      'lambda_': self.lambda_,
                      }
        config.update(new_config)
        return config

    @classmethod
    def from_config(cls, config):
        model = keras.layers.deserialize(config.pop("model"))
        features_metadata = config.pop("features_metadata")
        mixup = keras.layers.deserialize(config.pop("mixup"))
        cutmix = keras.layers.deserialize(config.pop("cutmix"))
        projection_head_1 = keras.layers.deserialize(config.pop("projection_head_1"))
        projection_head_2 = keras.layers.deserialize(config.pop("projection_head_2"))
        reconstruction_head = keras.layers.deserialize(config.pop("reconstruction_head"))
        return cls(model=model,
                   features_metadata=features_metadata,
                   mixup=mixup,
                   cutmix=cutmix,
                   projection_head_1=projection_head_1,
                   projection_head_2=projection_head_2,
                   reconstruction_head=reconstruction_head,
                   **config)
