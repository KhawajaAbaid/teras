import jax
import keras
from teras.backend.common.models.pretrainers.tab_transformer import (
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
        loss = self.compute_loss(y_true=x * mask,
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
        loss = self.compute_loss(y_true=mask,
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
