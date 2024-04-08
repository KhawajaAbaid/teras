import tensorflow as tf
import keras
from teras._src.backend.common.models.pretrainers.tab_transformer import (
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
        missing_rate: float, percentage of features to make missing.
            Defaults to 0.3 (or 30%)
        mask_seed: int, seed for generating mask. Defaults to 1337
    """
    def __init__(self,
                 model: keras.Model,
                 data_dim: int,
                 missing_rate: float = 0.3,
                 mask_seed: int = 1337,
                 **kwargs):
        super().__init__(model=model,
                         data_dim=data_dim,
                         missing_rate=missing_rate,
                         mask_seed=mask_seed,
                         **kwargs)

    def _create_mask(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(
                "Expected `input_shape` to have a rank of 2. "
                f"Received, {input_shape} with rank {len(input_shape)}")
        batch_size, _ = input_shape
        num_features_to_miss = int(self.missing_rate * self.data_dim)
        mask = tf.ones((num_features_to_miss,), dtype=tf.int32)
        mask = tf.pad(mask,
                      [[0, self.data_dim - num_features_to_miss]])
        mask = tf.tile(tf.expand_dims(mask, 0), (batch_size, 1))
        mask = tf.map_fn(lambda x: tf.random.shuffle(x,
                                                     seed=self.mask_seed),
                         mask,
                         dtype=tf.int32)
        return mask

    def train_step(self, data):
        mask = self._create_mask(ops.shape(data))
        mask = ops.cast(mask, data.dtype)
        # In mask, 1 indicates that the feature will be missing, while 0
        # indicates the opposite
        with tf.GradientTape() as tape:
            y_pred = self(data, mask=(1 - mask))
            loss = self.compute_loss(y=data * mask,
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
        replace_rate: float, percentage of features to replace.
            Defaults to 0.3 (or 30%)
        mask_seed: int, seed for generating mask. Defaults to 1337
    """
    def __init__(self,
                 model: keras.Model,
                 data_dim: int,
                 replace_rate: float = 0.3,
                 mask_seed: int = 1337,
                 **kwargs):
        super().__init__(model=model,
                         data_dim=data_dim,
                         replace_rate=replace_rate,
                         mask_seed=mask_seed,
                         **kwargs)

    def _create_mask(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(
                "Expected `input_shape` to have a rank of 2. "
                f"Received, {input_shape} with rank {len(input_shape)}")
        batch_size, _ = input_shape
        num_features_to_replace = int(self.replace_rate * self.data_dim)
        mask = tf.ones((num_features_to_replace,), dtype=tf.int32)
        mask = tf.pad(mask,
                      [[0, self.data_dim - num_features_to_replace]])
        mask = tf.tile(tf.expand_dims(mask, 0), (batch_size, 1))
        mask = tf.map_fn(lambda x: tf.random.shuffle(x,
                                                     seed=self.mask_seed),
                         mask,
                         dtype=tf.int32)
        return mask

    def train_step(self, data):
        mask = self._create_mask(ops.shape(data))
        mask = ops.cast(mask, data.dtype)
        # In mask, 1 indicates that the feature will be missing, while 0
        # indicates the opposite
        with tf.GradientTape() as tape:
            y_pred = self(data, mask=(1 - mask))
            loss = self.compute_loss(y=mask,
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


