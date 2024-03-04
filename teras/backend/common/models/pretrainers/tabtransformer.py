import keras
import numpy as np
from keras import random, ops, backend
from teras.layers.layer_list import LayerList


class BaseTabTransformerMLMPretrainer(keras.Model):
    def __init__(self,
                 model: keras.Model,
                 data_dim: int,
                 k: float = 0.3,
                 mask_seed: int = 1337,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.data_dim = data_dim
        if k < 0. or k > 1.0:
            raise ValueError(
                f"`k` must be in range [0, 1]. Received {k}"
            )
        self.k = k
        self.mask_seed = mask_seed
        self._numpy_rng = np.random.default_rng(self.mask_seed)

        self.features_predictor = keras.layers.Dense(
            units=data_dim,
            name="features_predictor")

        self._pretrained = False

    def build(self, input_shape):
        self.model.build(input_shape)

    def compile(self,
                loss=keras.losses.BinaryCrossentropy(),
                **kwargs):
        super().compile(loss=loss, **kwargs)

    def call(self, inputs, mask, **kwargs):
        x = inputs * mask
        x = self.model(x)
        x = self.features_predictor(x)
        x = x * (1 - mask)
        return x

    def _create_mask(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(
                "Expected `input_shape` to have a rank of 2. "
                f"Received, {input_shape} with rank {len(input_shape)}")
        batch_size, num_features = input_shape
        num_features_to_replace = self.k * num_features
        mask = np.zeros(input_shape)
        features_idx = np.arange(num_features)

        for i in range(batch_size):
            features_idx_to_replace = self._numpy_rng.choice(
                features_idx,
                num_features_to_replace,
                replace=False)
            mask[i, features_idx_to_replace] = 1.
        mask = backend.convert_to_tensor(mask, dtype="int32")
        return mask

    def get_config(self):
        config = {
            "name": self.name,
            "trainable": self.trainable,
            "model": keras.layers.serialize(self.model),
            "data_dim": self.data_dim,
            "k": self.k,
            "mask_seed": self.mask_seed
        }
        return config

    @classmethod
    def from_config(cls, config):
        model = keras.layers.deserialize(config.pop("model"))
        return cls(model=model, **config)

    @property
    def pretrained_model(self):
        if not self._pretrained:
            raise AssertionError(
                "The `fit` method of the `TabTransformerMLMPretrainer` "
                "has not yet been called. Please train the it before "
                "accessing the `pretrained_model` attribute."
            )
        return self.model


class BaseTabTransformerRTDPretrainer(keras.Model):
    def __init__(self,
                 model: keras.Model,
                 data_dim: int,
                 k: float = 0.3,
                 mask_seed: int = 1337,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        # here `k` represents replace rate instead of missing rate like
        # in the MLM version of the pretrainer
        if k < 0. or k > 1.0:
            raise ValueError(
                f"`k` must be in range [0, 1]. Received {k}"
            )
        self.k = k
        self.mask_seed = mask_seed
        self._numpy_rng = np.random.default_rng(self.mask_seed)
        self._seed_for_shuffling = random.SeedGenerator(1337)

        # binary predictors, one for each feature, which predict if the
        # value of the features is replaced or not
        self.predictors = LayerList([
            keras.layers.Dense(1,  name=f"feature_{i}_predictor")
            for i in range(data_dim)
        ],
            sequential=False)

    def build(self, input_shape):
        self.model.build(input_shape)

    def call(self, inputs, mask, **kwargs):
        # Since in RTD, for a sample, we randomly replace k% of its
        # features values using random values of those features.
        # We can efficiently achieve this by first getting
        # x_rand = shuffle(inputs)
        # then, to apply replacement,
        # inputs = (inputs * (1-mask)) + (x_rand * mask)
        inputs_shuffled = random.shuffle(inputs,
                                         axis=0,
                                         seed=self._seed_for_shuffling)
        inputs = inputs * mask + (inputs_shuffled * (1 - mask))
        x = self.model(inputs)
        predictions = self.predictors[0](x)
        for i in range(1, self.data_dim):
            pred = self.predictors[i](x)
            predictions = ops.concatenate([predictions, pred], axis=1)
        return predictions

    def _create_mask(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(
                "Expected `input_shape` to have a rank of 2. "
                f"Received, {input_shape} with rank {len(input_shape)}")
        batch_size, num_features = input_shape
        num_features_to_replace = self.k * num_features
        mask = np.zeros(input_shape)
        features_idx = np.arange(num_features)

        for i in range(batch_size):
            features_idx_to_replace = self._numpy_rng.choice(
                features_idx,
                num_features_to_replace,
                replace=False)
            mask[i, features_idx_to_replace] = 1.
        mask = backend.convert_to_tensor(mask, dtype="int32")
        return mask

    def get_config(self):
        config = {
            "name": self.name,
            "trainable": self.trainable,
            "model": keras.layers.serialize(self.model),
            "data_dim": self.data_dim,
            "k": self.k,
            "mask_seed": self.mask_seed
        }
        return config

    @classmethod
    def from_config(cls, config):
        model = keras.layers.deserialize(config.pop("model"))
        return cls(model=model, **config)

    @property
    def pretrained_model(self):
        if not self._pretrained:
            raise AssertionError(
                "The `fit` method of the `TabTransformerRTDPretrainer` "
                "has not yet been called. Please train the it before "
                "accessing the `pretrained_model` attribute."
            )
        return self.model
