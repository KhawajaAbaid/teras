import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from teras.models.gain import Generator, Discriminator, GAIN
from teras.losses.pcgain import generator_loss, generator_pretraining_loss, discriminator_loss
from teras.utils.pcgain import cluster
from typing import List, Tuple, Union


LIST_OR_TUPLE = Union[List[int], Tuple[int]]
INT_OR_FLOAT = Union[int, float]
HIDDEN_BLOCK_TYPE = Union[keras.layers.Layer, keras.models.Model]


class Classifier(keras.Model):
    """
    The auxiliary classifier for the PC-GAIN architecture
    proposed by Yufeng Wang et al. in the paper
    "PC-GAIN: Pseudo-label Conditional Generative Adversarial
    Imputation Networks for Incomplete Data"

    Reference(s):
        https://arxiv.org/abs/2011.07770

    Args:
        data_dim: `int`,
            The dimensionality of the input dataset.
            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.
        num_classes: `int`,
            Number of classes to predict.
            It should be equal to the `num_clusters`,
            computed during the pseudo label generation.
        units_values: `List[int]` or `Tuple[int]`,
            A list/tuple of units to construct hidden block of classifier.
            For each element, a new hidden layer will be added.
            By default, `units_values` = [`data_dim`, `data_dim`]
        activation_hidden: default "relu",
            Activation function to use for the hidden layers in the classifier.
        activation_out: default "softmax",
            Activation function to use for the output layer of classifier.
    """
    def __init__(self,
                 data_dim: int,
                 num_classes: int = None,
                 units_values: LIST_OR_TUPLE = None,
                 activation_hidden="relu",
                 activation_out="softmax",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.units_values = units_values
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out

        if self.units_values is None:
            self.units_values = [data_dim] * 2

        self.hidden_block = models.Sequential(name="classifier_hidden_block")
        for units in self.units_values:
            self.hidden_block.add(layers.Dense(units,
                                               activation=self.activation_hidden))

        self.output_layer = layers.Dense(self.num_classes,
                                         activation=self.activation_out)

    def call(self, inputs):
        x = self.hidden_block(inputs)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        new_config = {'data_dim': self.data_dim,
                      'num_classes': self.num_classes,
                      'units_values': self.units_values,
                      'activation_hidden': self.activation_hidden,
                      'activation_out': self.activation_out
                      }
        config.update(new_config)
        return config


class PCGAIN(keras.Model):
    """
    PC-GAIN is a missing data imputation model based
    on the GAIN architecture.

    This implementation is based on the architecture
    proposed by Yufeng Wang et al. in the paper
    "PC-GAIN: Pseudo-label Conditional Generative Adversarial
    Imputation Networks for Incomplete Data"

    Reference(s):
        https://arxiv.org/abs/2011.07770

    Args:
        data_dim: `int`,
            The dimensionality of the input dataset.
            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.
       generator_units_values: `List[int]` or `Tuple[int]`,
            A list or tuple of units for constructing hidden block for the Generator.
            For each value, a `GAINDiscriminatorBlock`
            (`from teras.layers import GAINGeneratorBlock`)
            of that dimensionality (units) is added to the generator to form the
            `hidden block` of the generator.
            By default, generator_units_values = [data_dim, data_dim].
       discriminator_units_values: `List[int]` or `Tuple[int]`,
           A list or tuple of units for constructing hidden block for the Discriminator.
           For each value, a `GAINDiscriminatorBlock`
           (`from teras.layers import GAINDiscriminatorBlock`)
           of that dimensionality (units) is added to the discriminator
           to form the `hidden block` of the discriminator.
           By default, discriminator_units_values = [data_dim, data_dim].
       generator_activation_hidden: default "relu",
           Activation function to use for the hidden layers in the hidden block
           for the Generator.
       discriminator_activation_hidden: default "relu",
           Activation function to use for the hidden layers in the hidden block
           for the Discriminator.
       generator_activation_out: default "sigmoid",
           Activation function to use for the output layer of the Generator.
       discriminator_activation_out: default "sigmoid",
           Activation function to use for the output layer of the Discriminator.
       num_discriminator_steps: `int`, default 1,
           Number of discriminator training steps per PCGAIN training step.
       hint_rate: `float`, default 0.9,
           Hint rate will be used to sample binary vectors for
           `hint vectors` generation. Should be between 0. and 1.
           Hint vectors ensure that generated samples follow the
           underlying data distribution.
       alpha: `float`, default 200.,
           Hyper parameter for the generator loss computation that
           controls how much weight should be given to the MSE loss.
           Precisely, `generator_loss` = `cross_entropy_loss` + `alpha` * `mse_loss`
           The higher the `alpha`, the more the mse_loss will affect the
           overall generator loss.
       beta: `float`, default 100.,
           Hyper parameter for generator loss computation that
           controls the contribution of the classifier's loss to the
           overall generator loss.
               num_clusters: `int`, default 5,
       Number of clusters to cluster the imputed data
           that is generated during pretraining.
           These clusters serve as pseudo labels for training of classifier.
       clustering_method: `str`, default "kmeans",
           Should be one of the following,
           ["Agglomerative", "KMeans", "MiniBatchKMeans", "Spectral", "SpectralBiclustering"]
           The names are case in-sensitive.
    """
    def __init__(self,
                 data_dim: int,
                 generator_units_values: LIST_OR_TUPLE = None,
                 discriminator_units_values: LIST_OR_TUPLE = None,
                 generator_activation_hidden="relu",
                 discriminator_activation_hidden="relu",
                 generator_activation_out="sigmoid",
                 discriminator_activation_out="sigmoid",
                 num_discriminator_steps: int = 1,
                 hint_rate: float = 0.9,
                 alpha: float = 200.,
                 beta: float = 100.,
                 num_clusters: int = 5,
                 clustering_method: str = "kmeans",
                 **kwargs):
        super().__init__(**kwargs)
        if not 0. <= hint_rate <= 1.0:
            raise ValueError("`hint_rate` should be between 0. and 1. "
                             f"Received {hint_rate}.")
        self.data_dim = data_dim
        self.generator_units_values = generator_units_values
        self.discriminator_units_values = discriminator_units_values
        self.generator_activation_hidden = generator_activation_hidden
        self.discriminator_activation_hidden = discriminator_activation_hidden
        self.generator_activation_out = generator_activation_out
        self.discriminator_activation_out = discriminator_activation_out
        self.num_discriminator_steps = num_discriminator_steps
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.beta = beta
        self.num_clusters = num_clusters
        self.clustering_method = clustering_method

        # self.generator = Generator(data_dim=self.data_dim,
        #                            units_values=self.generator_units_values,
        #                            activation_hidden=self.generator_activation_hidden,
        #                            activation_out=self.generator_activation_out)
        # self.discriminator = Discriminator(data_dim=self.data_dim,
        #                                    units_values=self.discriminator_units_values,
        #                                    activation_hidden=self.discriminator_activation_hidden,
        #                                    activation_out=self.discriminator_activation_out)

        # Since we pretrain using the EXACT SAME architecture as GAIN
        # so here we simply use the GAIN model, which acts as `pretrainer`.
        # And since it instantiates generators and discriminator models,
        # we can pass these generator and discriminator parameters to GAIN
        # which acts as a proxy pretrainer and have it instantiate and pretrain them,
        # and then we can access those pretrained Generator and Discriminator models
        # and use them here in our PC-GAIN architecture.
        self.pretrainer = GAIN(data_dim=self.data_dim,
                               generator_units_values=self.generator_units_values,
                               generator_activation_hidden=self.generator_activation_hidden,
                               generator_activation_out=self.generator_activation_out,
                               discriminator_units_values=self.discriminator_units_values,
                               discriminator_activation_hidden=self.discriminator_activation_hidden,
                               discriminator_activation_out=self.discriminator_activation_out,
                               hint_rate=self.hint_rate,
                               alpha=self.alpha
                                )
        self.classifier = Classifier(data_dim=self.data_dim,
                                     num_classes=self.num_clusters)

        self.z_sampler = tfp.distributions.Uniform(low=0.,
                                                   high=0.01,
                                                   name="z_sampler")
        self.hint_vectors_sampler = tfp.distributions.Binomial(total_count=1,
                                                               probs=self.hint_rate,
                                                               name="hint_vectors_generator")

        # Flag to keep track of whether generator and discriminator
        # have been pretrained or not
        self._pretrained = False
        # Flag to check for training start and call specific functions
        self._first_batch = True

        # Loss trackers
        self.generator_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    def compile(self,
                generator_optimizer=keras.optimizers.Adam(),
                discriminator_optimizer=keras.optimizers.Adam(),
                generator_loss=generator_loss,
                discriminator_loss=discriminator_loss,
                generator_pretraining_loss=generator_pretraining_loss,
                classifier_loss=keras.losses.SparseCategoricalCrossentropy(),
                classifier_optimizer=keras.optimizers.RMSprop(),
                ):
        super().compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.generator_pretraining_loss = generator_pretraining_loss
        # compile the pretrainer (which is GAIN model)
        # using these losses.
        # Note that only generator has a different pretraining loss,
        # as the discriminator uses the same loss!
        self.pretrainer.compile(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator_loss=generator_pretraining_loss,
                                discriminator_loss=discriminator_loss
                                )

        self.classifier.compile(loss=classifier_loss,
                                optimizer=classifier_optimizer)

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def call(self, inputs, mask=None, training=None):
        if mask is not None:
            inputs = tf.concat([inputs, mask], axis=1)
        return self.generator(inputs)

    def pretrain(self,
                 pretraining_dataset=None,
                 pretrainer_fit_kwargs: dict = {},
                 classifier_fit_kwargs: dict = {}):
        """
        Pretrain function that does everything required before PC-GAIN training.
        It does the following in the given order:
            1. Pretrains the Generator and Discriminator and imputes pretraining dataset.
            2. Using clustering on imputed data, generates pseudo labels.
            3. Trains the classifier.

        Terminology Note:
            `Pretrainer` refers to the model used to pretrain Discriminator and Generator.
            `Classifier` refers to the model trained on imputed data and pseudo labels.

        Args:
            pretraining_dataset: Get it from DataSampler's `get_pretraining_dataset` method.
            pretrainer_fit_kwargs: Dictionary of keyword arguments for pretrainer's fit method.
            classifier_fit_kwargs: Dictionary of keyword arguments for classifier's fit method.

        p.s. I know accepting arguments as dict is not a good design but for the time being,
        please tolerate it. I'll fix it in the near future!
        """
        print("Pretrain Step 1/4: Pretraining the generator and discriminator")
        self.pretrainer.fit(pretraining_dataset, **pretrainer_fit_kwargs)
        print("\nPretrain Step 2/4: Imputing data")
        imputed_data = self.pretrainer.predict(pretraining_dataset)
        self.generator = self.pretrainer.get_generator()
        self.discriminator = self.pretrainer.get_discriminator()
        print("\nPretrain Step 3/4: Generating pseudo labels")
        pseudo_labels = cluster(x=imputed_data,
                                num_clusters=self.num_clusters,
                                method=self.clustering_method)
        print("\nPretrain Step 4/4: Training the classifier")
        self.classifier.fit(imputed_data, y=pseudo_labels, **classifier_fit_kwargs)
        print("\nPretraining completed successfully!")
        self._pretrained = True
        print("\nTraining PCGAIN")

    # We'll utilize the same DataTransformer and DataSampler classes from
    # the GAIN architecture. and hence the incoming data follows the same format.
    def train_step(self, data):
        if self._first_batch:
            if not self._pretrained:
                raise AssertionError("The model has not yet been pretrained. "
                                     "PC-GAIN requires Generator and Discriminator "
                                     "to be pretrained before the main training. "
                                     "You must call the `pretrain` method before "
                                     "calling the `fit` method.")
            self._first_batch = False

        # data is a tuple of x_generator and x_discriminator batches
        # drawn from the dataset. The reason behind generating two separate
        # batches of data at each step is that PCGAIN requires separate
        # batches of data for Discriminator and Generator.
        x_gen, x_disc = data

        # ====> Training the Discriminator <=====
        for _ in range(self.num_discriminator_steps):
            # Create mask
            mask = tf.constant(1.) - tf.cast(tf.math.is_nan(x_disc), dtype=tf.float32)
            # replace nans with 0.
            x_disc = tf.where(tf.math.is_nan(x_disc), x=0., y=x_disc)
            # Sample noise
            z = self.z_sampler.sample(sample_shape=tf.shape(x_disc))
            # Sample hint vectors
            hint_vectors = self.hint_vectors_sampler.sample(sample_shape=(tf.shape(x_disc)))
            hint_vectors *= mask
            # Combine random vectors with original data
            x_disc = x_disc * mask + (1 - mask) * z
            # Generate samples
            generated_samples = self(x_disc, mask=mask)
            # Combine generated samples with original data
            x_hat_disc = (generated_samples * (1 - mask)) + (x_disc * mask)
            with tf.GradientTape() as tape:
                discriminator_pred = self.discriminator(tf.concat([x_hat_disc, hint_vectors], axis=1))
                loss_disc = self.discriminator_loss(discriminator_pred, mask)
            gradients = tape.gradient(loss_disc, self.discriminator.trainable_weights)
            self.discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_weights))

        # =====> Train the Generator <=====
        mask = tf.constant(1.) - tf.cast(tf.math.is_nan(x_gen), dtype=tf.float32)
        x_gen = tf.where(tf.math.is_nan(x_gen), x=0., y=x_gen)
        z = self.z_sampler.sample(sample_shape=tf.shape(x_gen))
        hint_vectors = self.hint_vectors_sampler.sample(sample_shape=(tf.shape(x_gen)))
        hint_vectors *= mask
        x_gen = x_gen * mask + (1 - mask) * z
        # "Introducing Pseudo label supervision"
        # we don't need to use one hot, since we're using
        # softmax activation at the end and it gives us
        # probability for each class
        with tf.GradientTape() as tape:
            generated_samples = self(x_gen, mask=mask)
            # Combine generated samples with original/observed data
            x_hat = (generated_samples * (1 - mask)) + (x_gen * mask)
            classifier_preds = self.classifier(x_hat)
            disc_preds = self.discriminator(tf.concat([x_hat, hint_vectors], axis=1))
            loss_gen = self.generator_loss(generated_samples=generated_samples,
                                           real_samples=x_gen,
                                           discriminator_pred=disc_preds,
                                           mask=mask,
                                           alpha=self.alpha,
                                           beta=self.beta,
                                           classifier_pred=classifier_preds)
        gradients = tape.gradient(loss_gen, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))

        # Update custom tracking metrics
        self.generator_loss_tracker.update_state(loss_gen)
        self.discriminator_loss_tracker.update_state(loss_disc)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"generator_" + m.name: m.result() for m in self.generator.metrics})
        results.update({"discriminator_" + m.name: m.result() for m in self.discriminator.metrics})
        return results

    def predict_step(self, data):
        """
        Args:
            Transformed data.
        Returns:
            Imputed data that should be reverse transformed
            to its original form.
        """
        if isinstance(data, tuple):
            data = data[0]
        data = tf.cast(data, tf.float32)
        # Create mask
        mask = tf.constant(1.) - tf.cast(tf.math.is_nan(data), dtype=tf.float32)
        data = tf.where(tf.math.is_nan(data), x=0., y=data)
        # Sample noise
        z = self.z_sampler.sample(sample_shape=tf.shape(data))
        x = mask * data + (1 - mask) * z
        imputed_data = self(x, mask=mask)
        imputed_data = mask * data + (1 - mask) * imputed_data
        return imputed_data

    def impute(self,
               x,
               data_transformer=None,
               reverse_transform=True,
               batch_size=None,
               verbose="auto",
               steps=None,
               callbacks=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False
               ):
        """
        Impute function combines PC-GAIN's `predict` method and
        DataTransformer's `reverse_transform` method to fill
        the missing data and transform into the original format.
        It exposes all the arguments taken by the `predict` method.

        Args:
            x: `pd.DataFrame`, dataset with missing values.
            data_transformer: An instance of DataTransformer class which
                was used in transforming the raw input data
            reverse_transform: `bool`, default True, whether to reverse
                transformed the raw imputed data to original format.

        Returns:
            Imputed data in the original format.
        """
        x_imputed = self.predict(x,
                                 batch_size=batch_size,
                                 verbose=verbose,
                                 steps=steps,
                                 callbacks=callbacks,
                                 max_queue_size=max_queue_size,
                                 workers=workers,
                                 use_multiprocessing=use_multiprocessing)
        if reverse_transform:
            if data_transformer is None:
                raise ValueError("To reverse transform the raw imputed data, `data_transformer` must not be None. "
                                 "Please pass the instance of DataTransformer class used to transform the input "
                                 "data as argument to this `impute` method."
                                 "Or alternatively, you can set `reverse_transform` parameter to False, "
                                 "and manually reverse transform the generated raw data to original format "
                                 "using the `reverse_transform` method of DataTransformer instance.")
            x_imputed = data_transformer.reverse_transform(x_imputed)
        return x_imputed

    def get_config(self):
        config = super().get_config()
        config.update({"data_dim": self.data_dim,
                       "generator_units_values": self.generator_units_values,
                       "discriminator_units_values": self.discriminator_units_values,
                       "generator_activation_hidden": self.generator_activation_hidden,
                       "discriminator_activation_hidden": self.discriminator_activation_hidden,
                       "generator_activation_out": self.generator_activation_out,
                       "discriminator_activation_out": self.discriminator_activation_out,
                       "hint_rate": self.hint_rate,
                       "alpha": self.alpha,
                       "beta": self.beta,
                       "num_clusters": self.num_clusters,
                       "clustering_method": self.clustering_method,
                       })
        return config
