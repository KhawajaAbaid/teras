import jax
import jax.numpy as jnp
import keras
from teras._src.backend.common.models.pretrainers.tab_transformer import (
    BaseTabTransformerMLMPretrainer,
    BaseTabTransformerRTDPretrainer
)
from keras import ops


class TabTransformerMLMPretrainer(BaseTabTransformerMLMPretrainer):
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
        key = jax.random.PRNGKey(self.mask_seed)
        num_features_to_miss = int(self.missing_rate * self.data_dim)
        mask = jnp.ones((num_features_to_miss,), dtype=jnp.int32)
        mask = jnp.pad(mask,
                       (0, self.data_dim - num_features_to_miss))
        mask = jnp.tile(jnp.expand_dims(mask, 0), (batch_size, 1))
        mask = jax.random.permutation(key, mask, axis=1, independent=True)
        return mask

    def compute_loss_and_updates(self,
                                 trainable_variables,
                                 non_trainable_variables,
                                 x,
                                 mask,
                                 training=False):
        y_pred, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            mask=(1 - mask),
            training=training
        )
        loss = self.compute_loss(y=x * mask,
                                 y_pred=y_pred)
        return loss, (y_pred, non_trainable_variables)

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables
        ) = state
        mask = self._create_mask(ops.shape(data))
        # In mask, 1 indicates that the feature will be missing, while 0
        # indicates the opposite
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates,
                                     has_aux=True)
        (loss, (y_pred, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            data,
            mask=mask,
            training=True
        )
        (
            trainable_variables,
            optimizer_variables
        ) = self.optimizer.stateless_apply(
            optimizer_variables,
            grads,
            trainable_variables
        )

        # Update metrics
        logs = {}
        new_metric_variables = []
        for metric in self.metrics:
            this_metric_variables = metrics_variables[
                                    len(new_metric_variables): len(
                                        new_metric_variables) + len(
                                        metric.variables)
                                    ]
            if metric.name == "loss":
                this_metric_variables = metric.stateless_update_state(
                    this_metric_variables,
                    loss
                )
            else:
                this_metric_variables = metric.stateless_update_state(
                    this_metric_variables,
                    data * mask,
                    y_pred
                )
            logs[metric.name] = metric.stateless_result(
                this_metric_variables)
            new_metric_variables += this_metric_variables

        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metric_variables
        )
        return logs, state


class TabTransformerRTDPretrainer(BaseTabTransformerRTDPretrainer):
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
        key = jax.random.PRNGKey(self.mask_seed)
        num_features_to_replace = int(self.replace_rate * self.data_dim)
        mask = jnp.ones((num_features_to_replace,), dtype=jnp.int32)
        mask = jnp.pad(mask,
                       (0, self.data_dim - num_features_to_replace))
        mask = jnp.tile(jnp.expand_dims(mask, 0), (batch_size, 1))
        mask = jax.random.permutation(key, mask, axis=1, independent=True)
        return mask

    def compute_loss_and_updates(self,
                                 trainable_variables,
                                 non_trainable_variables,
                                 x,
                                 mask,
                                 training=False):
        y_pred, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            mask=(1 - mask),
            training=training
        )
        loss = self.compute_loss(y=mask,
                                 y_pred=y_pred)
        return loss, (y_pred, non_trainable_variables)

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables
        ) = state

        mask = self._create_mask(ops.shape(data))
        # In mask, 1 indicates that the feature will be missing, while 0
        # indicates the opposite
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates,
                                     has_aux=True)
        (loss, (y_pred, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            data,
            mask=mask,
            training=True
        )
        (
            trainable_variables,
            optimizer_variables
        ) = self.optimizer.stateless_apply(
            optimizer_variables,
            grads,
            trainable_variables
        )

        # Update metrics
        logs = {}
        new_metric_variables = []
        for metric in self.metrics:
            this_metric_variables = metrics_variables[
                                    len(new_metric_variables): len(
                                        new_metric_variables) + len(
                                        metric.variables)
                                    ]
            if metric.name == "loss":
                this_metric_variables = metric.stateless_update_state(
                    this_metric_variables,
                    loss
                )
            else:
                this_metric_variables = metric.stateless_update_state(
                    this_metric_variables,
                    data * mask,
                    y_pred
                )
            logs[metric.name] = metric.stateless_result(
                this_metric_variables)
            new_metric_variables += this_metric_variables

        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metric_variables
        )
        return logs, state
