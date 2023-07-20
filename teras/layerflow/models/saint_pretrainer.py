import tensorflow as tf
from tensorflow import keras
from teras.losses.saint import info_nce_loss, denoising_loss


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

        mixup: ``keras.layers.Layer``,
            An instance of ``MixUp`` layer or any custom layer that can work
            in its place.
            You can import the ``MixUp`` layer as follows,
                >>> from teras.layerflow.layers import MixUp

        cutmix: ``keras.layers.Layer``,
            An instance of ``CutMix`` layer or any custom layer that can work
            in its place.
            You can import the `CutMix` layer as follows,
                >>> from teras.layerflow.layers import CutMix

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

        self.features_metadata = self.model.features_metadata
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

        # Since we want to keep the encode value to what originally was,
        # which is also stored in the model's encode_categorical_values attribute, so we set it equal to that
        self.model.categorical_feature_embedding.encode = self.model.encode_categorical_values
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
                      'mixup': keras.layers.serialize(self.mixup),
                      'cutmix': keras.layers.serialize(self.cutmix),
                      'projection_head_1': keras.layers.serialize(self.projection_head_1),
                      'projection_head_2': keras.layers.serialize(self.projection_head_2),
                      'temperature': self.temperature,
                      'lambda_': self.lambda_,
                      }
        config.update(new_config)
        return config

    @classmethod
    def from_config(cls, config):
        model = keras.layers.deserialize(config.pop("model"))
        mixup = keras.layers.deserialize(config.pop("mixup"))
        cutmix = keras.layers.deserialize(config.pop("cutmix"))
        projection_head_1 = keras.layers.deserialize(config.pop("projection_head_1"))
        projection_head_2 = keras.layers.deserialize(config.pop("projection_head_2"))
        reconstruction_head = keras.layers.deserialize(config.pop("reconstruction_head"))
        return cls(model=model,
                   mixup=mixup,
                   cutmix=cutmix,
                   projection_head_1=projection_head_1,
                   projection_head_2=projection_head_2,
                   reconstruction_head=reconstruction_head,
                   **config)
