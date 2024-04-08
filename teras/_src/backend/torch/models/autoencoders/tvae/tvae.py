import torch
import keras
from keras import ops
from teras._src.backend.common.models.autoencoders.tvae.tvae import BaseTVAE


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
        self.zero_grad()
        generated_samples, sigmas, mean, log_var = self(data)
        loss = self.compute_loss(real_samples=data,
                                 generated_samples=generated_samples,
                                 sigmas=sigmas,
                                 mean=mean,
                                 log_var=log_var)
        loss.backward()
        grads = [v.value.grad for v in self.trainable_variables]
        with torch.no_grad():
            self.optimizer.apply(grads,
                                 self.trainable_variables)
        sigmas = ops.clip(sigmas, x_min=0.01, x_max=1.0)
        self.decoder.sigmas = sigmas
        self.loss_tracker.update_state(loss)
        logs = {m.name: m.result() for m in self.metrics}
        return logs
