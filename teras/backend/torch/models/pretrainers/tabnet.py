import torch
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

    def train_step(self, data):
        # Sample mask
        mask = random.binomial(
            shape=ops.shape(data),
            counts=1,
            probabilities=self.missing_feature_probability)
        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()

        # Call
        reconstructed = self(data, mask)

        # Compute loss
        loss = self._reconstruction_loss_fn(
            real=data,
            reconstructed=reconstructed,
            mask=mask
        )

        # Compute gradients
        loss.backward()
        trainable_variables = [v for v in self.trainable_variables]
        gradients = [v.value.grad for v in trainable_variables]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(
                grads=gradients,
                trainable_variables=trainable_variables,
            )

        # Update metrics
        for metric in self.metrics:
            if (metric.name == "loss" or
                    metric.name == "reconstruction_loss"):
                metric.update_state(loss)
            else:
                metric.update_state(data, reconstructed)

        logs = {m.name: m.result() for m in self.metrics}
        return logs
