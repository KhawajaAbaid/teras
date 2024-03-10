import keras
from keras import random, ops
from teras.layers.cutmix import CutMix
from teras.layers.mixup import MixUp
from teras.layers.saint.embedding import SAINTEmbedding
from teras.layers.saint.projection_head import SAINTProjectionHead
from teras.layers.saint.reconstruction_head import SAINTReconstructionHead


class BaseSAINTPretrainer(keras.Model):
    def __init__(self,
                 model: keras.Model,
                 cardinalities: list,
                 embedding_dim: int,
                 cutmix_probability: float = 0.3,
                 mixup_alpha: float = 1.,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.cardinalities = cardinalities
        self.embedding_dim = embedding_dim
        self.cutmix_probability = cutmix_probability
        self.mixup_alpha = mixup_alpha
        self.seed = seed

        self.cutmix = CutMix(probability=self.cutmix_probability,
                             mask_seed=self.seed,
                             shuffle_seed=self.seed + 10)
        self.mixup = MixUp(alpha=self.mixup_alpha,
                           lambda_seed=self.seed + 20,
                           shuffle_seed=self.seed + 30)
        self.embedding = SAINTEmbedding(
            embedding_dim=self.embedding_dim,
            cardinalities=self.cardinalities)

        # For the computation of contrastive loss, we use projection heads.
        # Projection head hidden dimensions as calculated by the
        # official implementation
        data_dim = len(cardinalities)
        projection_head_hidden_dim = 6 * embedding_dim * data_dim // 5
        projection_head_output_dim = embedding_dim * data_dim // 2
        self.projection_head_original = SAINTProjectionHead(
            hidden_dim=projection_head_hidden_dim,
            output_dim=projection_head_output_dim,
            name="projection_head_original"
        )
        self.projection_head_mixed = SAINTProjectionHead(
            hidden_dim=projection_head_hidden_dim,
            output_dim=projection_head_output_dim,
            name="projection_head_mixed"
        )
        self.flatten = keras.layers.Flatten()

        self.reconstruction_head = SAINTReconstructionHead(
            cardinalities=self.cardinalities,
            name="reconstruction_head"
        )

        self._pretrained = False
        self._constrastive_loss_tracker = keras.metrics.Mean(
            name="constrastive_loss"
        )
        self._denoising_loss_tracker = keras.metrics.Mean(
            name="denoising_loss"
        )

    def build(self, input_shape):
        self.embedding.build(input_shape)
        input_shape = self.embedding.compute_output_shape(input_shape)
        self.model.build(input_shape)
        input_shape = self.model.compute_output_shape(input_shape)
        self.projection_head_original.build(input_shape)
        self.projection_head_mixed.build(input_shape)
        self.reconstruction_head.build(input_shape)

    def compile(self,
                contrastive_loss=None,
                denoising_loss=None,
                **kwargs):
        super().compile(**kwargs)
        self.contrastive_loss = contrastive_loss
        self.denoising_loss = denoising_loss

    def call(self, inputs, **kwargs):
        original = inputs
        # Apply cutmix to raw inputs
        mixed = self.cutmix(inputs)

        # Apply embedding layer to raw inputs as well as cutmixed inputs
        original = self.embedding(original)
        mixed = self.embedding(mixed)

        # Apply mixup in embedding space
        mixed = self.mixup(mixed)

        # Pass these through encoder
        original = self.model(original)
        mixed = self.model(mixed)

        # For contrastive loss
        z_original = self.projection_head_original(original)
        z_original = self.flatten(z_original)
        z_mixed = self.projection_head_mixed(mixed)
        z_mixed = self.flatten(z_mixed)

        # For denoising loss
        reconstructed = self.reconstruction_head(mixed)

        return (z_original, z_mixed), reconstructed

    @property
    def pretrained_embedding_layer(self):
        if not self._pretrained:
            raise AssertionError(
                "The `fit` method of the `SAINTPretrainer` has not yet "
                "been called. Please train the it before accessing the "
                "`pretrained_embedding_layer` attribute."
            )
        return self.embedding

    @property
    def pretrained_model(self):
        if not self._pretrained:
            raise AssertionError(
                "The `fit` method of the `TabTransformerMLMPretrainer` "
                "has not yet been called. Please train the it before "
                "accessing the `pretrained_model` attribute."
            )
        return self.model

    def get_config(self):
        config = {
            "name": self.name,
            "trainable": self.trainable,
            "model": keras.layers.serialize(self.model),
            "cardinalities": self.cardinalities,
            "cutmix_probability": self.cutmix_probability,
            "mixup_alpha": self.mixup_alpha,
            "seed": self.seed
        }
        return config

    @classmethod
    def from_config(cls, config):
        model = keras.layers.deserialize(config.pop("model"))
        return cls(model=model, **config)

