import tensorflow as tf
from tensorflow import keras
from teras.losses import VimeSelfSupervisedLoss


@keras.saving.register_keras_serializable(package="teras.layerflow.models")
class VimeSelf(keras.Model):
    """
    Self-supervised learning part of the VIME architecture,
    proposed by Jinsung Yoon et al. in the paper
    "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        input_dim: ``int``,
            The dimensionality of the input dataset,
            or the number of features in the dataset.

        encoder: ``keras.layers.Layer``,
            An instance of the ``VimeEncoder`` layer or any custom
            layer that can work in its place.
            You can import the ``VimeEncoder`` layer as follows,
                >>> from teras.layers import VimeEncoder

        mask_estimator: ``keras.layers.Layer``,
            An instance of the ``VimeMaskEstimator`` layer or any custom
            layer that can work in its place.
            You can import the ``VimeMaskEstimator`` layer as follows,
                >>> from teras.layers import VimeMaskEstimator

        feature_estimator: ``keras.layers.Layer``,
            An instance of the ``VimeFeatureEstimator`` layer or any custom
            layer that can work in its place.
            You can import the ``VimeFeatureEstimator`` layer as follows,
                >>> from teras.layers import VimeFeatureEstimator
    """
    def __init__(self,
                 input_dim: int,
                 encoder: keras.layers.Layer = None,
                 mask_estimator: keras.layers.Layer = None,
                 feature_estimator: keras.layers.Layer = None,
                 **kwargs
                 ):

        if encoder is None:
            raise ValueError("`encoder` cannot be None. You must pass an instance of `VimeEncoder` or any custom layer "
                             "that can work in its place. \n"
                             "You can import the `VimeEncoder` as `from teras.layers import VimeEncoder`")

        if mask_estimator is None:
            raise ValueError("`mask_estimator` cannot be None. You must pass an instance of `VimeMaskEstimator` or any "
                             "custom layer that can work in its place. \n"
                             "You can import the `VimeMaskEstimator` as `from teras.layers import VimeMaskEstimator`")

        if feature_estimator is None:
            raise ValueError("`feature_estimator` cannot be None. You must pass an instance of `VimeFeatureEstimator` "
                             "or any custom layer that can work in its place. \n"
                             "You can import the `VimeFeatureEstimator` as "
                             "`from teras.layers import VimeFeatureEstimator`")

        inputs = keras.layers.Input(shape=(input_dim,))
        # Encoding using VimeEncoder
        encoded_inputs = encoder(inputs)
        # Mask estimator
        mask_out = mask_estimator(encoded_inputs)
        # Feature estimator
        feature_out = feature_estimator(encoded_inputs)
        # Calling model.get_layer("encode") returns error that the encoder layer is not connected
        # to any node. To combat that issue, instead of calling layers one by one on inputs
        # directly, we instead make a functional model and call it on inputs.
        # Making a functional model allows us to create a Input layer and connect it with
        # the encoder layer.
        super().__init__(inputs=inputs,
                         outputs={"mask_estimator": mask_out,
                                  "feature_estimator": feature_out},
                         **kwargs)
        self.input_dim = input_dim
        self.encoder = encoder
        self.mask_estimator = mask_estimator
        self.feature_estimator = feature_estimator

    def compile(self,
                loss={"mask_estimator": keras.losses.MeanSquaredError(),
                      "feature_estimator": keras.losses.BinaryCrossentropy()},
                loss_weights={"mask_estimator": 1.0,
                              "feature_estimator": 2.0},
                **kwargs):
        super().compile(loss=loss,
                        loss_weights=loss_weights,
                        **kwargs)

    def get_encoder(self):
        """
        Retrieves the encoder part of the trained model
        Returns:
            Encoder part of the model
        """
        # Extract encoder part
        encoder = keras.models.Model(inputs=self.input, outputs=self.encoder.output)
        return encoder

    def get_config(self):
        config = super().get_config()
        config.update({"input_dim": self.input_dim,
                       "encoder": keras.layers.serialize(self.encoder),
                       "mask_estimator": keras.layers.serialize(self.mask_estimator),
                       "feature_estimator": keras.layers.serialize(self.feature_estimator)
                       }
                      )
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        encoder = keras.layers.deserialize(config.pop("encoder"))
        mask_estimator = keras.layers.deserialize(config.pop("mask_estimator"))
        feature_estimator = keras.layers.deserialize(config.pop("feature_estimator"))
        return cls(input_dim=input_dim,
                   encoder=encoder,
                   mask_estimator=mask_estimator,
                   feature_estimator=feature_estimator,
                   **config)


@keras.saving.register_keras_serializable(package="teras.layerflow.models")
class VimeSemi(keras.Model):
    """
    Semi-supervised learning part of the VIME architecture,
    proposed by Jinsung Yoon et al. in the paper
    "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset.

        pretrained_encoder: ``keras.Model``,
            Pretrained instance of ``VimEncoder``, pretrained during the 
            Self-Supervised learning phase.
            You can access it after training the ``VimeSelf`` model through its
            ``.get_encoder()`` method.

        mask_generation_and_corruption: ``keras.layers.Layer``,
            An instance oof the ``VimeMaskGenerationAndCorruption`` layer,
            or any custom layer that can work in its place.
            You can import the ``VimeMaskGenerationAndCorruption`` layer as follows,
                >>> from teras.layers import VimeMaskGenerationAndCorruption

        predictor: ``keras.layers.Layer``,
            An instance oof the ``VimePredictor`` layer,
            or any custom layer that can work in its place.
            You can import the ``VimePredictor`` layer as follows,
                >>> from teras.layers import VimePredictor

        K: ``int``, default 3,
            Number of augmented samples

        beta: ``float``, default 1.0,
            Hyperparameter to control supervised and unsupervised loss
    """
    def __init__(self,
                 input_dim: int,
                 pretrained_encoder: keras.Model = None,
                 mask_generation_and_corruption: keras.layers.Layer = None,
                 predictor: keras.layers.Layer = None,
                 K: int = 3,
                 beta: float = 1.0,
                 **kwargs):
        if pretrained_encoder is None:
            raise ValueError("`pretrained_encoder` cannot be None. You must pass an instance of pretrained encoder "
                             "trained during the self supervised phase. \n"
                             "You can access it after training the `VimeSelf` model through its "
                             "`.get_encoder()` method.")

        if mask_generation_and_corruption is None:
            raise ValueError("`mask_generation_and_corruption` cannot be None. You must pass an instance of "
                             "`VimeMaskGenerationAndCorruption` layer or any "
                             "custom layer that can work in its place. \n"
                             "You can import the `VimeMaskGenerationAndCorruption` as "
                             "`from teras.layers import VimeMaskGenerationAndCorruption`")

        if predictor is None:
            raise ValueError("`predictor` cannot be None. You must pass an instance of `VimePredictor` or any "
                             "custom layer that can work in its place. \n"
                             "You can import the `VimePredictor` as `from teras.layers import VimePredictor`")

        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.pretrained_encoder = pretrained_encoder
        self.K = K
        self.beta = beta

        self.mask_corrupt_block = keras.models.Sequential()
        self.input_layer = keras.layers.Input(shape=(self.input_dim,))
        self.mask_corrupt_block.add(self.input_layer)
        self.mask_generation_and_corruption = mask_generation_and_corruption
        self.mask_corrupt_block.add(self.mask_and_corrupt)

        self.predictor = predictor
        self.self_supervised_loss = VimeSelfSupervisedLoss()
        self.add_loss(self.self_supervised_loss)

    def train_step(self, data):
        labeled_dataset, unlabeled_dataset = data
        X_batch = labeled_dataset["x_labeled"]
        targets = labeled_dataset["y_labeled"]
        X_unlabeled_batch = unlabeled_dataset["x_unlabeled"]

        # Encode labeled data
        X_batch_encoded = self.pretrained_encoder(X_batch,
                                                  training=False)

        X_unlabeled_batch_encoded = list()
        for _ in range(self.K):
            X_unlabeled_batch_temp = self.mask_corrupt_block(X_unlabeled_batch)

            # Encode Corrupted Samples
            X_unlabeled_batch_temp = self.pretrained_encoder(X_unlabeled_batch_temp,
                                                             training=False)
            X_unlabeled_batch_encoded = X_unlabeled_batch_encoded + [X_unlabeled_batch_temp]

        # Convert list to tensor matrix
        X_unlabeled_batch_encoded = tf.stack(X_unlabeled_batch_encoded, axis=0)

        with tf.GradientTape() as tape:
            outputs = self({"labeled": X_batch_encoded,
                            "unlabeled": X_unlabeled_batch_encoded},
                           training=True)
            yu_loss = self.self_supervised_loss(y_true=None, y_pred=outputs['yv_hat_logit'])
            y_loss = self.compiled_loss(targets, outputs['y_hat_logit'])
            loss = y_loss + self.beta * yu_loss
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(targets, outputs['y_hat'])
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Since at the time of prediction / evaluation we won't be processing any unlabeled data
        # so we get rid of all the unlabeled data logic / processing
        inputs, targets = data
        X_batch = inputs
        # Encode labeled data
        X_batch_encoded = self.pretrained_encoder(X_batch,
                                                  training=False)
        outputs = self({"labeled": X_batch_encoded},
                       training=False)
        self.compiled_loss(targets, outputs['y_hat_logit'])
        self.compiled_metrics.update_state(targets, outputs['y_hat'])
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=None):
        X = inputs["labeled"]
        outputs = {}
        y_hat_logit, y_hat = self.predictor(X)
        outputs['y_hat_logit'] = y_hat_logit
        outputs['y_hat'] = y_hat
        # We process unlabeled data only while training
        if training:
            X_unlabled = inputs["unlabeled"]
            yv_hat_logit, yv_hat = self.predictor(X_unlabled)
            outputs['yv_hat_logit'] = yv_hat_logit
            outputs['yv_hat'] = yv_hat
        outputs['beta'] = self.beta
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'input_dim': self.input_dim,
                       'pretrained_encoder': keras.layers.serialize(self.pretrained_encoder),
                       'mask_generation_and_corruption': keras.layers.serialize(self.mask_generation_and_corruption),
                       'predictor': keras.layers.serialize(self.predictor),
                       'K': self.K,
                       'beta': self.beta,
                       }
                      )
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        pretrained_encoder = keras.layers.deserialize(config.pop("pretrained_encoder"))
        mask_generation_and_corruption = keras.layers.deserialize(config.pop("mask_generation_and_corruption"))
        predictor = keras.layers.deserialize(config.pop("predictor"))
        return cls(input_dim=input_dim,
                   pretrained_encoder=pretrained_encoder,
                   mask_generation_and_corruption=mask_generation_and_corruption,
                   predictor=predictor,
                   **config)
