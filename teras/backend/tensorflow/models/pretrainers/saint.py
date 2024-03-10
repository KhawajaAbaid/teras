import tensorflow as tf
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
        with tf.GradientTape() as tape:
            (z_real, z_mixed), reconstructed = self.model(data)
            c_loss = self.contrastive_loss(z_real,
                                           z_mixed,
                                           self.temperature,
                                           self.lambda_c)
            d_loss = self.denoising_loss(data,
                                         reconstructed)
            loss = c_loss + self.lambda_ * d_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(gradients,
                             self.trainable_variables)

        for metric in self.metrics:
            if metric.name == "constrastive_loss":
                metric.update_state(c_loss)
            elif metric.name == "denoising_loss":
                metric.update_state(d_loss)
            elif metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(data, reconstructed)

        logs = {m.name: m.result() for m in self.metrics}
        return logs
