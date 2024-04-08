import keras
from keras import random, ops
from teras._src.layers.cutmix import CutMix
from teras._src.layers.mixup import MixUp
from teras._src.layers.saint.embedding import SAINTEmbedding
from teras._src.layers.saint.projection_head import SAINTProjectionHead
from teras._src.layers.saint.reconstruction_head import SAINTReconstructionHead
import numpy as np


class BaseSAINTPretrainer(keras.Model):
    def __init__(self,
                 model: keras.Model,
                 cardinalities: list,
                 embedding_dim: int,
                 cutmix_probability: float = 0.3,
                 mixup_alpha: float = 1.,
                 temperature: float = 0.7,
                 lambda_: float = 10.,
                 lambda_c: float = 0.5,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.cardinalities = cardinalities
        self.embedding_dim = embedding_dim
        self.cutmix_probability = cutmix_probability
        self.mixup_alpha = mixup_alpha
        self.temperature = temperature
        self.lambda_ = lambda_
        self.lambda_c = lambda_c
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
        self.projection_head_real = SAINTProjectionHead(
            hidden_dim=projection_head_hidden_dim,
            output_dim=projection_head_output_dim,
            name="projection_head_real"
        )
        self.projection_head_mixed = SAINTProjectionHead(
            hidden_dim=projection_head_hidden_dim,
            output_dim=projection_head_output_dim,
            name="projection_head_mixed"
        )
        self.flatten = keras.layers.Flatten()

        self.reconstruction_head = SAINTReconstructionHead(
            cardinalities=self.cardinalities,
            embedding_dim=self.embedding_dim,
            name="reconstruction_head"
        )

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.constrastive_loss_tracker = keras.metrics.Mean(
            name="constrastive_loss"
        )
        self.denoising_loss_tracker = keras.metrics.Mean(
            name="denoising_loss"
        )

        self._pretrained = False

    def build(self, input_shape):
        self.cutmix.build(input_shape)
        self.mixup.build(input_shape)
        self.embedding.build(input_shape)
        input_shape = self.embedding.compute_output_shape(input_shape)
        self.model.build(input_shape)
        input_shape = self.model.compute_output_shape(input_shape)
        self.projection_head_real.build(input_shape)
        self.projection_head_mixed.build(input_shape)
        self.flatten.build(input_shape)
        self.reconstruction_head.build(input_shape)

    def compute_loss(self, x, x_reconstructed, z, z_augmented, cardinalities,
                     temperature=0.7, lambda_=10., lambda_c=0.5):
        """

        Args:
            x: Samples drawn from the original dataset.
            x_reconstructed: Samples reconstructed by the reconstruction head.
            z: Encodings of the real samples
            z_augmented: Encodings of the augmented samples
            cardinalities: list, a list cardinalities of all the features
                in the dataset in the same order as the features' occurrence.
                For numerical features, use the value `0` as indicator at
                the corresponding index.
                You can use the `compute_cardinalities` function from
                `teras.utils` package for this purpose.
            temperature: float, used in the computation of the
                `contrastive_loss` to scale logits. Defaults to 0.7
            lambda_: float, acts as a weight when adding the contrastive loss
                and the denoising loss together.
                `loss = constrastive_loss + lambda_ * denoising_loss`
                Defaults to 10.
            lambda_c: float, used in the computation of the contrastive
                loss. Similar to `lambda_` is helps combined two sub-losses
                within the contrastive loss. Defaults to 0.5

        Returns:
            Combined contrastive and denoising loss.
        """
        # ====================
        # Constrastive loss
        # ====================
        batch_size = ops.shape(z)[0]
        labels = ops.arange(batch_size)
        logits_ab = ops.matmul(z, ops.transpose(z_augmented)) / temperature
        logits_ba = ops.matmul(z_augmented, ops.transpose(z)) / temperature
        loss_a = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
            y_true=labels,
            y_pred=logits_ab)
        loss_b = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
            y_true=labels,
            y_pred=logits_ba)
        c_loss = lambda_c * (loss_a + loss_b) / 2

        # ==================
        # Denoising loss
        # ==================
        num_features = len(cardinalities)
        categorical_features_idx = np.flatnonzero(cardinalities)
        num_categorical_features = len(categorical_features_idx)
        categorical_cardinalities = np.asarray(cardinalities)[
            categorical_features_idx]
        total_categories = sum(cardinalities)
        d_loss = 0.
        # The reconstructed samples have dimensions equal to number of
        # categorical features + number of categories in all the categorical
        # features combined.
        # In other words, each categorical feature gets expanded into
        # `number of categories` features due to softmax layer during the
        # reconstruction phase.
        # Since the order they occur in is still the same
        # except for that the first `total_categories` number of features are
        # categorical and following are continuous
        for current_idx, feature_idx in enumerate(categorical_features_idx):
            num_categories = categorical_cardinalities[current_idx]
            d_loss += keras.losses.SparseCategoricalCrossentropy(
                from_logits=True)(
                # real_samples have the original feature order
                y_true=x[:, feature_idx],
                y_pred=x_reconstructed[:, current_idx: current_idx + num_categories]
            )
            current_idx += 1
        # Check if there are numerical features -- if so, compute the mse loss
        mse_loss = ops.cond(pred=num_categorical_features < num_features,
                            true_fn=lambda: ops.sum(
                                keras.losses.MeanSquaredError()(
                                    x[:, num_categorical_features:],
                                    x_reconstructed[:, total_categories:])),
                            false_fn=lambda: 0.
                            )
        d_loss += mse_loss
        loss = c_loss + lambda_ * d_loss
        return loss, c_loss, d_loss

    def reset_metrics(self):
        self.loss_tracker.reset_state()
        self.constrastive_loss_tracker.reset_state()
        self.denoising_loss_tracker.reset_state()

    @property
    def metrics(self):
        metrics = [self.loss_tracker,
                   self.constrastive_loss_tracker,
                   self.denoising_loss_tracker,
                   ]
        return metrics

    def call(self, inputs, **kwargs):
        real = inputs
        # Apply cutmix to raw inputs
        mixed = self.cutmix(inputs)

        # Apply embedding layer to raw inputs as well as cutmixed inputs
        real = self.embedding(real)
        mixed = self.embedding(mixed)

        # Apply mixup in embedding space
        mixed = self.mixup(mixed)

        # Pass these through encoder
        real = self.model(real)
        mixed = self.model(mixed)

        # For contrastive loss
        z_real = self.projection_head_real(real)
        z_real = self.flatten(z_real)
        z_mixed = self.projection_head_mixed(mixed)
        z_mixed = self.flatten(z_mixed)

        # For denoising loss
        reconstructed = self.reconstruction_head(mixed)

        return (z_real, z_mixed), reconstructed

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
            "temperature": self.temperature,
            "lambda_": self.lambda_,
            "lambda_c": self.lambda_c,
            "seed": self.seed
        }
        return config

    @classmethod
    def from_config(cls, config):
        model = keras.layers.deserialize(config.pop("model"))
        return cls(model=model, **config)

