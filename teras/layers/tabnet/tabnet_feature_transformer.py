from tensorflow import keras
from teras.layers.tabnet.tabnet_feature_transformer_block import TabNetFeatureTransformerBlock


@keras.saving.register_keras_serializable("teras.layers.tabnet")
class TabNetFeatureTransformer(keras.layers.Layer):
    """
    Feature Transformer as proposed by Sercan et al. in TabNet paper.
    It is made up of ``FeatureTransformerBlock`` building blocks.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        units: ``int``, default 32,
            the dimensionality of the hidden representation in feature transformation block.
            Each layer first maps the representation to a ``2 * feature_transformer_dim``
            output and half of it is used to determine the
            non-linearity of the GLU activation where the other half is used as an
            input to GLU, and eventually ``feature_transformer_dim`` output is
            transferred to the next layer.

        num_shared_layers: ``int``, default 2.
            Number of shared layers to use in the Feature Transformer.
            These shared layers are `shared` across decision steps.

        num_decision_dependent_layers: ``int``, default 2.
            Number of decision dependent layers to use.
            In simple words, ``num_decision_dependent_layers`` are created for each
            decision step in the ``num_decision_steps``.
            For instance, if ``num_decision_steps = 5`` and  ``num_decision_dependent_layers = `2`
            then 10 layers will be created, 2 for each decision step.

        batch_momentum: ``float``, default 0.9,
            Momentum value to use for ``BatchNormalization`` layer.

        virtual_batch_size: ``int``, default 64,
            Batch size to use for ``virtual_batch_size`` parameter in ``BatchNormalization`` layer.
            This is typically much smaller than the `batch_size` used for training.
            And most importantly, the ``batch_size`` must always be divisible by the ``virtual_batch_size``.

        residual_normalization_factor: ``float``, default 0.5,
            In the feature transformer, except for the layer, every other layer utilizes
            normalized residuals, where ``residual_normalization_factor`` determines the scale of normalization.
    """
    shared_layers = None

    def __init__(self,
                 units: int = 32,
                 num_shared_layers: int = 2,
                 num_decision_dependent_layers: int = 2,
                 batch_momentum: float = 0.9,
                 virtual_batch_size: int = 64,
                 residual_normalization_factor: float = 0.5,
                 **kwargs):

        if num_shared_layers == 0 and num_decision_dependent_layers == 0:
            raise ValueError("Feature Transformer requires at least one of either shared or decision depenedent layers."
                             " But both `num_shared_layers` and `num_decision_dependent_layers` were passed a 0 value.")

        super().__init__(**kwargs)
        self.units = units
        self.num_shared_layers = num_shared_layers
        self.num_decision_dependent_layers = num_decision_dependent_layers
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.residual_normalization_factor = residual_normalization_factor

        # Since shared layers should be accessible across instances so we make them as class attributes
        # and use class method to instantiate them, the first time an instance of this FeatureTransformer
        # class is made.
        if self.shared_layers is None and num_shared_layers > 0:
            self._create_shared_layers(num_shared_layers,
                                       units,
                                       batch_momentum=batch_momentum,
                                       virtual_batch_size=virtual_batch_size,
                                       residual_normalization_factor=residual_normalization_factor)

        self.inner_block = keras.models.Sequential(name="inner_block")

        self.inner_block.add(self.shared_layers)

        if self.num_decision_dependent_layers > 0:
            decision_dependent_layers = keras.models.Sequential(name=f"decision_dependent_layers")
            for j in range(self.num_decision_dependent_layers):
                if j == 0 and self.num_shared_layers == 0:
                    # then it means that this is the very first layer in the Feature Transformer
                    # because no shard layers exist.
                    # hence we shouldn't use residual normalization
                    use_residual_normalization = False
                else:
                    use_residual_normalization = True
                decision_dependent_layers.add(TabNetFeatureTransformerBlock(
                    units=self.units,
                    batch_momentum=self.batch_momentum,
                    virtual_batch_size=self.virtual_batch_size,
                    use_residual_normalization=use_residual_normalization,
                    name=f"decision_dependent_layer_{j}")
                )

            self.inner_block.add(decision_dependent_layers)

    @classmethod
    def _create_shared_layers(cls, num_shared_layers, units, **kwargs):
        # The outputs of first layer in feature transformer isn't residual normalized
        # so we use a pointer to know which layer is first.
        # Important to keep in mind that shared layers ALWAYS precede the decision dependent layers
        # BUT we want to allow user to be able to use NO shared layers at all, i.e. 0.
        # Hence, only if the `num_shared_layers` is 0, should the first dependent layer for each step
        # be treated differently.
        # Initialize the block of shared layers
        cls.shared_layers = keras.models.Sequential(name="shared_layers")
        for i in range(num_shared_layers):
            if i == 0:
                # First layer should not use residual normalization
                use_residual_normalization = False
            else:
                use_residual_normalization = True
            cls.shared_layers.add(TabNetFeatureTransformerBlock(units=units,
                                                                use_residual_normalization=use_residual_normalization,
                                                                name=f"shared_layer_{i}",
                                                                **kwargs))

    @classmethod
    def reset_shared_layers(cls):
        """
        Resets the shared layers.

        You must reset shared layers, when you want to create
        new instances of ``TabNetFeatureTransformer`` for ``TabNetDecoder``
        after you have created those for the ``TabNetEncoder``.
        """
        cls.shared_layers = None

    def call(self, inputs):
        outputs = self.inner_block(inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units,
                       'num_shared_layers': self.num_shared_layers,
                       'num_decision_dependent_layers': self.num_decision_dependent_layers,
                       'batch_momentum': self.batch_momentum,
                       'virtual_batch_size': self.virtual_batch_size,
                       'residual_normalization_factor': self.residual_normalization_factor,
                       })
        return config
