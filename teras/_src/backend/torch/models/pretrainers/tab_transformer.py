import torch
import keras
from teras._src.backend.common.models.pretrainers.tab_transformer import (
    BaseTabTransformerMLMPretrainer,
    BaseTabTransformerRTDPretrainer
)
from keras import ops


class TabTransformerMLMPretrainer(BaseTabTransformerMLMPretrainer):
    def __init__(self,
                 model: keras.Model,
                 data_dim: int,
                 missing_rate: float = 0.3,
                 mask_seed: int = 1337,
                 **kwargs):
        super().__init__(model=model,
                         data_dim=data_dim,
                         missing_rate=missing_rate,
                         mask_seed=mask_seed,
                         **kwargs)

    def train_step(self, data):
        mask = self._create_mask(ops.shape(data))

        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()

        # In mask, 1 indicates that the feature will be missing, while 0
        # indicates the opposite
        y_pred = self(data, (1 - mask))
        loss = self.compute_loss(y=data * mask,
                                 y_pred=y_pred)
        # Run backwards pass
        loss.backward()
        # Get grads
        gradients = [v.value.grad for v in self.trainable_variables]
        # Optimize
        with torch.no_grad():
            self.optimizer.apply(
                gradients,
                self.trainable_variables
            )

        logs = {m.name: m.result() for m in self.metrics}
        return logs


class TabTransformerRTDPretrainer(BaseTabTransformerRTDPretrainer):
    def __init__(self,
                 model: keras.Model,
                 data_dim: int,
                 replace_rate: float = 0.3,
                 mask_seed: int = 1337,
                 **kwargs):
        super().__init__(model=model,
                         data_dim=data_dim,
                         replace_rate=replace_rate,
                         mask_seed=mask_seed,
                         **kwargs)

    def train_step(self, data):
        mask = self._create_mask(ops.shape(data))
        # In mask, 1 indicates that the feature will be missing, while 0
        # indicates the opposite
        y_pred = self(data, mask=(1 - mask))
        loss = self.compute_loss(y=mask,
                                 y_pred=y_pred)
        # Run backward pass
        loss.backward()
        # Get grads
        gradients = [v.value.grad for v in self.trainable_variables]
        # Optimize
        self.optimizer.apply(
            gradients,
            self.trainable_variables
        )

        logs = {m.name: m.result() for m in self.metrics}
        return logs
