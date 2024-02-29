import tensorflow as tf
import keras
from keras import ops, random
from teras.models.pretrainers.tabnet.pretrainer import BaseTabNetPretrainer


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
        if isinstance(data, tuple):
            data = data[0]
        # Sample mask
        mask = random.binomial(
            shape=ops.shape(data),
            counts=1,
            probabilities=self.missing_feature_probability)
        with tf.GradientTape() as tape:
            reconstructed = self(data, mask)
            loss = self._reconstruction_loss_fn(
                real=data,
                reconstructed=reconstructed,
                mask=mask
            )
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(
            grads=gradients,
            trainable_variables=self.trainable_variables
        )

        for metric in self.metrics:
            if (metric.name == "reconstruction_loss" or
                    metric.name == "loss"):
                metric.update_state(loss)
            else:
                metric.update_state(data, reconstructed)

        logs = {m.name: m.result() for m in self.metrics}
        return logs
