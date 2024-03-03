import jax
import keras
from keras import ops, random
from teras.backend.common.models.pretrainers.tabnet import BaseTabNetPretrainer


class TabNetPretrainer(BaseTabNetPretrainer):
    """
    TabNetPretrainer for pretraining `TabNetEncoder` as proposed by
    Arik et al. in the paper,
    "TabNet: Attentive Interpretable Tabular Learning"

    Args:
        encoder: keras.Model, instance of `TabNetEncoder` to pretrain
        decoder: keras.Model, instance of `TabNetDecoder`
        missing_feature_probability: float, probability of missing features
    """
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
        loss = self._reconstruction_loss_fn(real=x,
                                            reconstructed=reconstructed,
                                            mask=mask)
        return loss, (reconstructed, non_trainable_variables)

    def train_step(self, state, data):
        (trainable_variables,
         non_trainable_variables,
         optimizer_variables,
         metrics_variables
         ) = state
        # Grad fn
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates,
                                     has_aux=True)

        # Sample mask
        mask = random.binomial(
            shape=ops.shape(data),
            counts=1,
            probabilities=self.missing_feature_probability,
            seed=self._mask_seed_generator
        )

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
            optimizer_variables=optimizer_variables,
            grads=grads,
            trainable_variables=trainable_variables,
        )

        # Update metrics
        logs = {}
        new_metric_variables = []
        for metric in self.metrics:
            this_metric_variables = metrics_variables[
                len(new_metric_variables): len(new_metric_variables) + len(metric.variables)
            ]
            if (metric.name == "loss" or
                    metric.name == "reconstruction_loss"):
                this_metric_variables = metric.stateless_update_state(
                    this_metric_variables,
                    loss
                )
            else:
                this_metric_variables = metric.stateless_update_state(
                    this_metric_variables,
                    x,
                    reconstructed
                )
            logs[metric.name] = metric.stateless_result(this_metric_variables)
            new_metric_variables += this_metric_variables

        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metric_variables
        )
        return logs, state
