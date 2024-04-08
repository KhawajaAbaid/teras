import jax
import keras
from teras._src.backend.common.models.pretrainers.saint import BaseSAINTPretrainer


class SAINTPretrainer(BaseSAINTPretrainer):
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
        super().__init__(model=model,
                         cardinalities=cardinalities,
                         embedding_dim=embedding_dim,
                         cutmix_probability=cutmix_probability,
                         mixup_alpha=mixup_alpha,
                         temperature=temperature,
                         lambda_=lambda_,
                         lambda_c=lambda_c,
                         seed=seed,
                         **kwargs)

    def compute_loss_and_updates(self,
                                 trainable_variables,
                                 non_trainable_variables,
                                 x,
                                 cardinalities,
                                 temperature,
                                 lambda_,
                                 lambda_c,
                                 training=False):
        (
            ((z_real, z_mixed), reconstructed),
            non_trainable_variables
        ) = self.stateless_call(trainable_variables,
                                non_trainable_variables,
                                x,
                                training=training)
        loss, c_loss, d_loss = self.compute_loss(
            x=x, x_reconstructed=reconstructed, z=z_real, z_augmented=z_mixed,
            cardinalities=cardinalities, temperature=temperature,
            lambda_c=lambda_c, lambda_=lambda_,
        )
        return loss, (c_loss, d_loss, reconstructed,
                      non_trainable_variables)

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables
        ) = state

        grad_fn = jax.value_and_grad(self.compute_loss_and_updates,
                                     has_aux=True)

        (
            (loss, (c_loss, d_loss, reconstructed,
                    non_trainable_variables)),
            gradients
        ) = grad_fn(trainable_variables,
                    non_trainable_variables,
                    data,
                    self.cardinalities,
                    self.temperature,
                    self.lambda_,
                    self.lambda_c,
                    training=True)
        (
            trainable_variables,
            optimizer_variables
        ) = self.optimizer.stateless_apply(
            optimizer_variables,
            gradients,
            trainable_variables
        )

        # Update metrics
        logs = {}
        new_metric_variables = []
        for metric in self.metrics:
            this_metric_variables = metrics_variables[
                                    len(new_metric_variables): len(new_metric_variables) + len(metric.variables)
                                    ]
            if metric.name == "constrastive_loss":
                this_metric_variables = metric.stateless_update_state(
                    this_metric_variables,
                    c_loss
                )
            elif metric.name == "denoising_loss":
                this_metric_variables = metric.stateless_update_state(
                    this_metric_variables,
                    d_loss
                )
            elif metric.name == "loss":
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

        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metric_variables
        )
        return logs, state
