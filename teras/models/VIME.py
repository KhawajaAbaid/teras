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


class VimeSelf:
    """Self-supervised learning part in the paper
    "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"
    by Jinsung Yoon et al.

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        p_m: corruption probability
        alpha: hyper-parameter to control the weights of feature and mask losses
        encoder_activation: activation to use for encoder
        feature_estimator_activation: activation to use for feature estimator
        mask_estimator_activation: activation to use for mask estimator
        optimizer: optimizer to use for training the encoder
        feature_estimator_loss: loss to use for feature estimator
        mask_estimator_loss: loss to use for mask estimator

    Returns:
      encoder: Representation learning block
    """
    def __init__(self,
                 p_m,
                 alpha,
                 encoder_activation="relu",
                 feature_estimator_activation="sigmoid",
                 mask_estimator_activation="sigmoid",
                 feature_estimator_loss="mean_squared_error",
                 mask_estimator_loss="binary_crossentropy",
                 optimizer="rmsprop"
                 ):
        self.p_m = p_m
        self.alpha = alpha
        self.encoder_activation = encoder_activation
        self.feature_estimator_activation = feature_estimator_activation
        self.mask_estimator_activation = mask_estimator_activation
        self.feature_estimator_loss = feature_estimator_loss
        self.mask_estimator_loss = mask_estimator_loss
        self.optimizer = optimizer

        self.built = False

    def build(self, input_shape):
        """Builds layers and model"""
        _, dim = input_shape
        inputs = keras.layers.Input(shape=(dim,))
        # Encoder
        h = VimeEncoder(dim,
                        activation=self.encoder_activation,
                        name='encoder')(inputs)
        # Mask estimator
        mask_out = VimeMaskEstimator(dim,
                                    activation=self.mask_estimator_activation,
                                    name='mask_estimator')(h)
        # Feature estimator
        feature_out = VimeFeatureEstimator(dim,
                                           activation=self.feature_estimator_activation,
                                           name='feature_estimator')(h)
        self.model = models.Model(inputs=inputs,
                             outputs=[mask_out, feature_out])

        self.model.compile(optimizer=self.optimizer,
                           loss={'mask_estimator': self.mask_estimator_loss,
                                 'feature_estimator': self.feature_estimator_loss},
                           loss_weights={'mask_estimator': 1,
                                         'feature_estimator': self.alpha})

    def generate_corrupted_samples(self, x_unlabeled):
        """
        Generates corrupted samples
        Args:
            x_unlabeled: unlabeled dataset
        """
        m_unlabeled = vime_mask_generator(self.p_m,
                                    x_unlabeled)
        m_labeled, x_tilde = vime_pretext_generator(m_unlabeled, x_unlabeled)
        return m_labeled, x_tilde

    def fit(self,
            x_unlabeled,
            m_labeled=None,
            x_tilde=None,
            epochs=1,
            batch_size=None):
        """
        Trains the self supervising part, specifically encoder of VIME
        Args:
            x_unlabeled: unlabeled feature"""
        if not self.built:
            self.build(x_unlabeled.shape)
            self.built = True

        if m_labeled is None and x_tilde is None:
            m_labeled, x_tilde = self.generate_corrupted_samples(x_unlabeled)

        # Fit model on unlabeled data
        self.model.fit(x_tilde, {'mask_estimator': m_labeled, 'feature_estimator': x_unlabeled},
                       epochs=epochs,
                       batch_size=batch_size)
    def get_encoder(self):
        """
        Retrieves the encoder part of the trained model
        Returns:
            Encoder part of the model
        """
        # Extract encoder part
        encoder_layer = self.model.get_layer(name="encoder")
        encoder = models.Model(inputs=self.model.input, outputs=encoder_layer.output)
        return encoder


class PredictorTrainer(models.Model):
    """
    Trainer model to train the Predictor component of VIME Semi Supervised Framework.

    Reference(s):
        Figure 2 in VIME paper.
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        hidden_dim: Dimensionality of hidden layers. Also known as "Units" in Keras.
        input_dim: Dimensionality of input. If None, shape will be inferred at the time of training
        p_m: corruption probability
        K: number of augmented samples
        beta: hyperparameter to control supervised and unsupervised loss
        encoder: trained encoder
        n_labels: Number of labels
    """
    def __init__(self,
                 hidden_dim,
                 input_dim=None,
                 p_m=0.,
                 K=1,
                 beta=0.,
                 activation="relu",
                 encoder=None,
                 n_labels=1,
                 batch_size=128,
                 **kwargs):

        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.p_m = p_m
        self.K = K
        self.beta = beta
        self.activation = activation
        self.n_labels = n_labels
        self.encoder = encoder
        self.batch_size = batch_size

        self.mask_corrupt_block = keras.models.Sequential()
        self.input_layer = keras.layers.Input(shape=(input_dim,),
                                              batch_size=self.batch_size)
        self.mask_corrupt_block.add(self.input_layer)
        self.mask_and_corrupt = VimeMaskGenerationAndCorruption(self.p_m)
        self.mask_corrupt_block.add(self.mask_and_corrupt)
        self.predictor = VimePredictor(hidden_dim=hidden_dim,
                                       input_dim=self.input_dim,
                                       name="predictor",
                                       n_labels=self.n_labels,
                                       batch_size=self.batch_size)
        self.self_supervised_loss = VimeSelfSupervisedLoss()
        self.add_loss(self.self_supervised_loss)

    def train_step(self, data):
        inputs, targets = data
        X_batch = inputs["X"]
        X_unlabeled_batch = inputs["X_unlabeled"]

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
            outputs = self({"X": X_batch_encoded,
                            "X_unlabeled": X_unlabeled_batch_encoded},
                            training=True)
            yu_loss = self.self_supervised_loss(y_true=None, y_pred=outputs['yv_hat_logit'])
            y_loss = self.compiled_loss(targets, outputs['y_hat_logit'])
            loss = y_loss + self.beta * yu_loss
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(targets, outputs['y_hat'])

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, **kwargs):
        print("does it get called before or after??")
        X = inputs["X"]
        X_unlabled = inputs["X_unlabeled"]
        y_hat_logit, y_hat = self.predictor(X)
        yv_hat_logit, yv_hat = self.predictor(X_unlabled)
        return {'y_hat_logit': y_hat_logit,
                'y_hat': y_hat,
                'yv_hat_logit': yv_hat_logit,
                'yv_hat': yv_hat,
                'beta': self.beta}


class VimeSemi:
    def __init__(self,
                 hidden_dim,
                 p_m=0.3,
                 K=3,
                 beta=1.0,
                 encoder_file_path=None,
                 n_labels=1,
                 activation="relu",
                 optimizer="adam",
                 ):
        """
        Semi-supervied learning part in VIME.
        Args:
            hidden_dim: Dimensionality of hidden layers. Also known as "Units" in Keras.
            p_m: corruption probability
            K: number of augmented samples
            beta: hyperparameter to control supervised and unsupervised loss
            encoder_file_path: file path for the trained encoder function
            n_labels: Number of labels
        """
        self.hidden_dim = hidden_dim
        self.p_m = p_m
        self.K = K
        self.beta = beta
        self.encoder_file_path = encoder_file_path
        self.n_labels = n_labels
        custom_objects = {"Encoder": VimeEncoder}
        with keras.utils.custom_object_scope(custom_objects):
            self.encoder = keras.models.load_model(encoder_file_path)
        self.activation = activation
        self.optimizer = optimizer

    def train_the_predictor(self,
                                X,
                                y,
                                X_unlabeled,
                                validation_data=None,
                                validation_split=0.1,
                                batch_size=128,
                                epochs=1,
                                **kwargs):
        input_dim = X.shape[1]

        # work around for batch_dimension being None
        n_samples, _ = X.shape
        last_batch_size = batch_size
        num_batches = n_samples // batch_size
        if n_samples % batch_size != 0:
            last_batch_size = n_samples - (num_batches * batch_size)
            X = X[:-last_batch_size]
            X_unlabeled = X_unlabeled[:-last_batch_size]
            y = y[:-last_batch_size]
            warn(f"Number of samples in dataset isn't divisble by batch size. "
                 f"Last {last_batch_size} samples will be discarded.")

        predictor_trainer = PredictorTrainer(self.hidden_dim,
                                             input_dim=input_dim,
                                             p_m=self.p_m,
                                             K=self.K,
                                             beta=self.beta,
                                             activation=self.activation,
                                             encoder=self.encoder,
                                             n_labels=self.n_labels,
                                             batch_size=batch_size,)
        predictor_trainer.compile(optimizer=self.optimizer,
                                  loss = keras.losses.CategoricalCrossentropy(from_logits=True),
                                  metrics=[keras.metrics.CategoricalAccuracy()])
        predictor_trainer.fit({"X": X, "X_unlabeled": X_unlabeled},
                              y=y,
                              batch_size=batch_size,
                              validation_data=validation_data,
                              validation_split=validation_split,
                              epochs=epochs,
                              **kwargs)
        return predictor_trainer.predictor