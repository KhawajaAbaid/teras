import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from teras.layers import (VimeEncoder,
                          VimeMaskEstimator,
                          VimeFeatureEstimator,
                          VimePredictor,
                          VimeMaskGenerationAndCorruption)
from teras.utils import vime_mask_generator, vime_pretext_generator
from teras.losses import VimeSelfSupervisedLoss
from warnings import warn
import numpy as np


class VimeSelf(keras.Model):
    """Self-supervised learning part in the paper
    "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"
    by Jinsung Yoon et al.

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        p_m: corruption probability
        encoder_activation: activation to use for encoder
        feature_estimator_activation: activation to use for feature estimator
        mask_estimator_activation: activation to use for mask estimator
    """
    def __init__(self,
                 p_m: float = 0.3,
                 encoder_activation="relu",
                 feature_estimator_activation="sigmoid",
                 mask_estimator_activation="sigmoid",
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.p_m = p_m
        self.encoder_activation = encoder_activation
        self.feature_estimator_activation = feature_estimator_activation
        self.mask_estimator_activation = mask_estimator_activation

    def build(self, input_shape):
        """Builds layers and functional model"""
        _, input_dim= input_shape
        inputs = keras.layers.Input(shape=(input_dim,))
        # Encoder
        encoded_inputs = VimeEncoder(input_dim,
                                        input_shape=(input_dim,),
                                        activation=self.encoder_activation,
                                        name="encoder")(inputs)
        # Mask estimator
        mask_out = VimeMaskEstimator(input_dim,
                                    activation=self.mask_estimator_activation,
                                    name="mask_estimator")(encoded_inputs)
        # Feature estimator
        feature_out = VimeFeatureEstimator(input_dim,
                                           activation=self.feature_estimator_activation,
                                           name="feature_estimator")(encoded_inputs)
        # Calling model.get_layer("encode") returns error that the encoder layer is not connected
        # to any node. To combat that issue, instead of calling layers one by one on inputs
        # directly, we instead make a functional model and call it on inputs.
        # Making a functional model allows us to create a Input layer and connect it with
        # the encoder layer.
        self.functional_model = keras.Model(inputs=inputs,
                                            outputs={"mask_estimator": mask_out, "feature_estimator": feature_out},
                                            name="functional_model")

    def call(self, inputs):
        return self.functional_model(inputs)

    def get_encoder(self):
        """
        Retrieves the encoder part of the trained model
        Returns:
            Encoder part of the model
        """
        # Extract encoder part
        encoder_layer = self.functional_model.get_layer(name="encoder")
        encoder = models.Model(inputs=self.functional_model.input, outputs=encoder_layer.output)
        return encoder

    def get_config(self):
        config = super().get_config()
        new_config = {'p_m': self.p_m,
                      'encoder_activation': self.encoder_activation,
                      'feature_estimator_activation': self.feature_estimator_activation,
                      'mask_estimator_activation': self.mask_estimator_activation,
                      }
        config.update(new_config)
        return config


class VimeSemi(keras.Model):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 32,
                 p_m: float = 0.3,
                 K: int = 3,
                 beta: float = 1.0,
                 encoder_file_path=None,
                 num_labels: int = 1,
                 activation="relu",
                 batch_size: int = None,
                 **kwargs):
        """
        Semi-supervied learning part in VIME.
        Args:
            hidden_dim: Dimensionality of hidden layers. Also known as "Units" in Keras.
            p_m: corruption probability
            K: number of augmented samples
            beta: hyperparameter to control supervised and unsupervised loss
            encoder_file_path: file path for the trained encoder function
            num_labels: Number of labels
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p_m = p_m
        self.K = K
        self.beta = beta
        self.encoder_file_path = encoder_file_path
        self.num_labels = num_labels
        custom_objects = {"Encoder": VimeEncoder}
        with keras.utils.custom_object_scope(custom_objects):
            self.encoder = keras.models.load_model(encoder_file_path)
        self.activation = activation
        self.batch_size = batch_size
        self.custom_built = False

        self.mask_corrupt_block = keras.models.Sequential()
        self.input_layer = keras.layers.Input(shape=(self.input_dim,),
                                              batch_size=self.batch_size)
        self.mask_corrupt_block.add(self.input_layer)
        self.mask_and_corrupt = VimeMaskGenerationAndCorruption(self.p_m)
        self.mask_corrupt_block.add(self.mask_and_corrupt)
        self.predictor = VimePredictor(hidden_dim=self.hidden_dim,
                                       input_dim=self.input_dim,
                                       name="predictor",
                                       num_labels=self.num_labels,
                                       batch_size=self.batch_size)
        self.self_supervised_loss = VimeSelfSupervisedLoss()
        self.add_loss(self.self_supervised_loss)

    def train_step(self, data):
        labeled_dataset, unlabeled_dataset = data
        X_batch = labeled_dataset["x_labeled"]
        targets = labeled_dataset["y_labeled"]
        X_unlabeled_batch = unlabeled_dataset["x_unlabeled"]

        # Encode labeled data
        X_batch_encoded = self.encoder(X_batch,
                                       training=False)

        X_unlabeled_batch_encoded = list()
        for _ in range(self.K):
            X_unlabeled_batch_temp = self.mask_corrupt_block(X_unlabeled_batch)

            # Encode Corrupted Samples
            X_unlabeled_batch_temp = self.encoder(X_unlabeled_batch_temp,
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
        X_batch_encoded = self.encoder(X_batch,
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
        new_config = {'input_dim': self.input_dim,
                      'hidden_dim': self.hidden_dim,
                      'p_m': self.p_m,
                      'K': self.K,
                      'beta': self.beta,
                      'encoder_file_path': self.encoder_file_path,
                      'num_labels': self.num_labels,
                      'activation': self.activation,
                      'batch_size': self.batch_size,
                      }
        config.update(new_config)
        return config
