import jax
import keras
from keras.models import Model
from keras.src.saving import serialization_lib


class JAXGAN(Model):
    """
    Extends Keras.Model to support multiple optimizers for GAN based models.
    """
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 **kwargs):
        Model.__init__(self, **kwargs)
        self.generator = generator
        self.discriminator = discriminator

        # Loss trackers
        self.generator_loss_tracker = keras.metrics.Mean(
            name="generator_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(
            name="discriminator_loss")

    def compile(self,
                generator_optimizer=keras.optimizers.Adam(),
                discriminator_optimizer=keras.optimizers.Adam(),
                **kwargs
                ):
        super().compile(**kwargs)
        self.generator_optimizer = keras.optimizers.get(
            generator_optimizer)
        self.discriminator_optimizer = keras.optimizers.get(
            discriminator_optimizer)

    def get_compile_config(self):
        config = super().get_compile_config()
        config.update(
            serialization_lib.SerializableDict(
                generator_optimizer=self.generator_optimizer,
                discriminator_optimizer=self.discriminator_optimizer,
            )
        )
        return config.serialize()

    def compile_from_config(self, config):
        config = serialization_lib.deserialize_keras_object(config)
        generator_optimizer = config.pop("generator_optimizer")
        discriminator_optimizer = config.pop("discriminator_optimizer")
        self.compile(generator_optimizer=generator_optimizer,
                     discriminator_optimizer=discriminator_optimizer,
                     **config)
        if self.built:
            self.generator_optimizer.build(self.generator.trainable_variables)
            self.discriminator_optimizer.build(
                self.discriminator.trainable_variables)

    @property
    def optimizers(self):
        return [self.generator_optimizer, self.discriminator_optimizer]

    @property
    def optimizers_variables(self):
        vars = []
        for optimizer in self.optimizers:
            vars.extend(optimizer.variables)
        return vars

    def build_optimizers(self):
        if not self.built:
            raise AssertionError(
                "You must build the model before calling `build_optimizers()`."
            )
        self.generator_optimizer.build(self.generator.trainable_variables)
        self.discriminator_optimizer.build(
            self.discriminator.trainable_variables)

    @property
    def metrics(self):
        metrics = [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
        ]
        return metrics

    def compute_loss(self, **kwargs):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` doesn't provide an implementation for"
            f" the `compute_loss` method. Please use "
            f"`compute_discriminator_loss` or `compute_generator_loss` for "
            f"relevant purpose."
        )

    def call(self, **kwargs):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` doesn't provide an implementation "
            f"for the `call` method. Please use the call method of "
            f"`GAIN().generator` or `GAIN().discriminator`."
        )

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def get_config(self):
        config = super().get_config()
        config.update({
            'generator': keras.layers.serialize(self.generator),
            'discriminator': keras.layers.serialize(self.discriminator),
            'hint_rate': self.hint_rate,
            'alpha': self.alpha,
        })
        return config

    @classmethod
    def from_config(cls, config):
        generator = keras.layers.deserialize(config.pop("generator"))
        discriminator = keras.layers.deserialize(config.pop("discriminator"))
        return cls(generator=generator, discriminator=discriminator,
                   **config)

    def _get_jax_state(
        self,
        trainable_variables=False,
        non_trainable_variables=False,
        optimizer_variables=False,
        metrics_variables=False,
        purge_model_variables=False,
    ):
        state = []
        if trainable_variables:
            state.append([v.value for v in self.trainable_variables])
        if non_trainable_variables:
            state.append([v.value for v in self.non_trainable_variables])
        if optimizer_variables:
            state.append([v.value for v in self.optimizers_variables])
        if metrics_variables:
            state.append([v.value for v in self.metrics_variables])
        if purge_model_variables:
            self._purge_model_variables(
                trainable_variables=trainable_variables,
                non_trainable_variables=non_trainable_variables,
                optimizer_variables=optimizer_variables,
                metrics_variables=metrics_variables,
            )
        return tuple(state)

    def jax_state_sync(self):
        if not getattr(self, "_jax_state", None) or self._jax_state_synced:
            return

        trainable_variables = self._jax_state.get("trainable_variables", None)
        non_trainable_variables = self._jax_state.get(
            "non_trainable_variables", None
        )
        optimizer_variables = self._jax_state.get("optimizer_variables", None)
        metrics_variables = self._jax_state.get("metrics_variables", None)
        if trainable_variables:
            for ref_v, v in zip(self.trainable_variables, trainable_variables):
                ref_v.assign(v)
        if non_trainable_variables:
            for ref_v, v in zip(
                self.non_trainable_variables, non_trainable_variables
            ):
                ref_v.assign(v)
        if optimizer_variables:
            for ref_v, v in zip(self.optimizers_variables, optimizer_variables):
                ref_v.assign(v)
        if metrics_variables:
            for ref_v, v in zip(self.metrics_variables, metrics_variables):
                ref_v.assign(v)
        self._jax_state_synced = True

    def _record_training_state_sharding_spec(self):
        self._trainable_variable_shardings = [
            v.value.sharding for v in self.trainable_variables
        ]
        self._non_trainable_variable_shardings = [
            v.value.sharding for v in self.non_trainable_variables
        ]
        self._optimizer_variable_shardings = [
            v.value.sharding for v in self.optimizers_variables
        ]
        self._metrics_variable_shardings = [
            v.value.sharding for v in self.metrics_variables
        ]

    def _enforce_jax_state_sharding(
        self,
        trainable_variables=None,
        non_trainable_variables=None,
        optimizer_variables=None,
        metrics_variables=None,
    ):
        """Enforce the sharding spec constraint for all the training state.

        Since the output of the train/eval step will be used as inputs to next
        step, we need to ensure that they have the same sharding spec, so that
        jax.jit won't have to recompile the train/eval function.

        Note that this function will also rely on the recorded sharding spec
        for each of states.

        This function is expected to be called within the jitted train/eval
        function, especially around the end of the function.
        """
        trainable_variables = trainable_variables or []
        non_trainable_variables = non_trainable_variables or []
        optimizer_variables = optimizer_variables or []
        metrics_variables = metrics_variables or []

        for i in range(len(trainable_variables)):
            trainable_variables[i] = jax.lax.with_sharding_constraint(
                trainable_variables[i], self._trainable_variable_shardings[i]
            )
        for i in range(len(non_trainable_variables)):
            non_trainable_variables[i] = jax.lax.with_sharding_constraint(
                non_trainable_variables[i],
                self._non_trainable_variable_shardings[i],
            )
        for i in range(len(optimizer_variables)):
            optimizer_variables[i] = jax.lax.with_sharding_constraint(
                optimizer_variables[i], self._optimizer_variable_shardings[i]
            )
        for i in range(len(metrics_variables)):
            metrics_variables[i] = jax.lax.with_sharding_constraint(
                metrics_variables[i], self._metrics_variable_shardings[i]
            )
        return (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        )

    def _purge_model_variables(
        self,
        trainable_variables=False,
        non_trainable_variables=False,
        optimizer_variables=False,
        metrics_variables=False,
    ):
        """Remove all the model variable for memory saving.

        During JAX training, since the training function are stateless, we have
        to pass in and get the model weights over and over, during which the
        copy of the weights that attached to the KerasVariable are still and
        occupying extra memory. We remove those variable to save memory (for
        better memory utilization) at the beginning of the epoch, and reattach
        the value back to variables at the end of the epoch, via
        `jax_state_sync()`.
        """
        if trainable_variables:
            for v in self.trainable_variables:
                v._value = None
        if non_trainable_variables:
            for v in self.non_trainable_variables:
                v._value = None
        if optimizer_variables:
            for v in self.optimizer.variables:
                v._value = None
        if metrics_variables:
            for v in self.metrics_variables:
                v._value = None