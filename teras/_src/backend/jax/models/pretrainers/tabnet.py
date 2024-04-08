import jax
import keras
from keras import ops, random
from teras._src.backend.common.models.pretrainers.tabnet import BaseTabNetPretrainer


class TabNetPretrainer(BaseTabNetPretrainer):
    def __init__(self,
                 encoder: keras.Model,
                 decoder: keras.Model,
                 missing_feature_probability: float = 0.3,
                 **kwargs):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            missing_feature_probability=missing_feature_probability,
            **kwargs)

    def compute_loss_and_updates(self,
                                 trainable_variables,
                                 non_trainable_variables,
                                 x,
                                 mask,
                                 training=False):
        reconstructed, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            mask=mask,
            training=training,
        )
        loss = self.compute_loss(x=x,
                                 x_reconstructed=reconstructed,
                                 mask=mask)
        return loss, (reconstructed, non_trainable_variables)

    def train_step(self, state, data):
        (trainable_variables,
         non_trainable_variables,
         optimizer_variables,
         metrics_variables
         ) = state
        seed = non_trainable_variables[0]
        # Sample mask
        mask = random.binomial(
            shape=ops.shape(data),
            counts=1,
            probabilities=self.missing_feature_probability,
            seed=seed,
        )
        # Grad fn
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates,
                                     has_aux=True)

        # Compute gradients
        (loss, (reconstructed, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            data,
            mask,
            training=True
        )

        # Update weights and optimizer vars
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
                                    len(new_metric_variables): len(new_metric_variables) + len(metric.variables)
                                    ]
            if metric.name == "loss":
                this_metric_variables = metric.stateless_update_state(
                    this_metric_variables,
                    loss
                )
            else:
                this_metric_variables = metric.stateless_update_state(
                    this_metric_variables,
                    data,
                    reconstructed
                )
            logs[metric.name] = metric.stateless_result(this_metric_variables)
            new_metric_variables += this_metric_variables

        seed = jax.random.split(seed, 1)[0]
        non_trainable_variables[0] = seed
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metric_variables
        )
        return logs, state
