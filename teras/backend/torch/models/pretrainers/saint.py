import torch
import keras
from teras.backend.common.models.pretrainers.saint import BaseSAINTPretrainer


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

    def train_step(self, data):
        # Zero the grads, hmm.
        self.zero_grad()

        # Run the forward pass, just like that?
        (z_real, z_mixed), reconstructed = self.model(data)

        # Compute losses, not too bad!
        c_loss = self.contrastive_loss(z_real,
                                       z_mixed,
                                       self.temperature,
                                       self.lambda_c)
        d_loss = self.denoising_loss(data,
                                     reconstructed,
                                     self.cardinalities)
        loss = c_loss + self.lambda_ * d_loss

        # Backward pass, cool!
        loss.backward()

        # Compute grads, easy!
        gradients = [v.value.grad for v in self.trainable_variables]

        # Optimize, nice!
        with torch.no_grad():
            self.optimizer.apply(gradients,
                                 self.trainable_variables)

        # Update metrics, useful!
        for metric in self.metrics:
            if metric.name == "constrastive_loss":
                metric.update_state(c_loss)
            elif metric.name == "denoising_loss":
                metric.update_state(d_loss)
            elif metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(data, reconstructed)

        # Return logs, grape!
        logs = {m.name: m.result() for m in self.metrics}
        return logs
