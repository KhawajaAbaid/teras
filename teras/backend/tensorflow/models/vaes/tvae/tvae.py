import tensorflow as tf
import keras
from keras import ops
from teras.backend.common.models.vaes.tvae.tvae import BaseTVAE


class TVAE(BaseTVAE):
    def __init__(self,
                 encoder: keras.Model,
                 decoder: keras.Model,
                 metadata: dict,
                 data_dim: int,
                 latent_dim: int = 128,
                 loss_factor: float = 2.,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         metadata=metadata,
                         data_dim=data_dim,
                         latent_dim=latent_dim,
                         loss_factor=loss_factor,
                         seed=seed,
                         **kwargs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            generated_samples, sigmas, mean, log_var = self(data)
            loss = self.compute_loss(real_samples=data,
                                     generated_samples=generated_samples,
                                     sigmas=sigmas,
                                     mean=mean,
                                     log_var=log_var)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(grads,
                             self.trainable_variables)
        self.add_loss(loss)
        sigmas = ops.clip(sigmas, x_min=0.01, x_max=1.0)
        self.decoder.sigmas = sigmas
        self.loss_tracker.update_state(loss)
        logs = {m.name: m.result() for m in self.metrics}
        return logs
