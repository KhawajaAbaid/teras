import tensorflow as tf
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

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        # Sample mask
        mask = random.binomial(
            shape=ops.shape(data),
            counts=1,
            probabilities=self.missing_feature_probability,
            seed=self._seed_generator
        )
        with tf.GradientTape() as tape:
            reconstructed = self(data, mask, training=True)
            loss = self.compute_loss(x=data,
                                     x_reconstructed=reconstructed,
                                     mask=mask)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(
            grads=gradients,
            trainable_variables=self.trainable_variables
        )

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(data, reconstructed)

        logs = {m.name: m.result() for m in self.metrics}
        return logs
