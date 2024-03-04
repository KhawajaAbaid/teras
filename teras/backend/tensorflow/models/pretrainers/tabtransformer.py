import tensorflow as tf
import keras
from teras.backend.common.models.pretrainers.tabtransformer import (
    BaseTabTransformerMLMPretrainer,
    BaseTabTransformerRTDPretrainer
)
from keras import ops


class TabTransformerMLMPretrainer(BaseTabTransformerMLMPretrainer):
    """
    MLM version of pretrainer for TabTransformer.

    Args:
        model: keras.Model, instance of `TabTransformerBackbone` model to
            pretrain.
        data_dim: int, number of features in the dataset
        k: float, percentage of features to make missing.
            Defaults to 0.3 (or 30%)
        mask_seed: int, seed for generating mask. Defaults to 1337
    """
    def __init__(self,
                 model: keras.Model,
                 data_dim: int,
                 k: float = 0.3,
                 mask_seed: int = 1337,
                 **kwargs):
        super().__init__(model=model,
                         data_dim=data_dim,
                         k=k,
                         mask_seed=mask_seed,
                         **kwargs)

    def train_step(self, data):
        mask = self._create_mask(ops.shape(data))
        # In mask, 1 indicates that the feature will be missing, while 0
        # indicates the opposite
        with tf.GradientTape() as tape:
            y_pred = self(data, mask=(1 - mask))
            loss = self.compute_loss(y_true=data * mask,
                                     y_pred=y_pred)
        # Compute grads
        gradients = tape.gradient(loss, self.trainable_variables)
        # Optimize
        self.optimizer.apply(
            gradients,
            self.trainable_variables
        )

        logs = {m.name: m.result() for m in self.metrics}
        return logs


class TabTransformerRTDPretrainer(BaseTabTransformerRTDPretrainer):
    """
    RTD version of pretrainer for TabTransformer.

    Args:
        model: keras.Model, instance of `TabTransformerBackbone` model to
            pretrain.
        data_dim: int, number of features in the dataset
        k: float, percentage of features to replace.
            Defaults to 0.3 (or 30%)
        mask_seed: int, seed for generating mask. Defaults to 1337
    """
    def __init__(self,
                 model: keras.Model,
                 data_dim: int,
                 k: float = 0.3,
                 mask_seed: int = 1337,
                 **kwargs):
        super().__init__(model=model,
                         data_dim=data_dim,
                         k=k,
                         mask_seed=mask_seed,
                         **kwargs)

    def train_step(self, data):
        mask = self._create_mask(ops.shape(data))
        # In mask, 1 indicates that the feature will be missing, while 0
        # indicates the opposite
        with tf.GradientTape() as tape:
            y_pred = self(data, mask=(1 - mask))
            loss = self.compute_loss(y_true=mask,
                                     y_pred=y_pred)
        # Compute grads
        gradients = tape.gradient(loss, self.trainable_variables)
        # Optimize
        self.optimizer.apply(
            gradients,
            self.trainable_variables
        )

        logs = {m.name: m.result() for m in self.metrics}
        return logs


