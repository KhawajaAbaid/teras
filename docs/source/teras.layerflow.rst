LayerFlow API
=======================
Teras ``LayerFlow API`` maximizes the flexibility while minimizing the interface. That may sound a bit contradictory at
first, but let me explain.
Here, instead of specifying individual parameters specific to sub layers/models, the user instead specifies instances of
those sub layers, models or blocks that are used in the given model architecture.

For instance, instead of specifying the ``embedding_dim`` parameter value, the user specifies an instance of
``CategoricalFeatureEmbedding`` layer.
Now in this instance, we're just passing one parameter instead of another so it may not seem like much beneficial at
first glance but let me highlight how it can immensely help depending on your use case.

Since all you need to pass is an instance of layer, it can be any layer, there's no restriction that it must be an
instance of ``CategoricalFeatureEmbedding`` layer — which means that you get complete control over not just customizing
the existing layers offered by Teras but also you can design/create an ``Embedding`` layer of your own that can work in
the place of the original ``CategoricalFeatureEmbedding`` layer or any other layer for that matter.
This is especially useful, if you're a designing a new architecture and want to rapidly test out new modifications of
the existing architectures by just plugging certain custom layers of your own in place of the default ones.
Pretty cool, right?

In many cases, to reduce the plethora of parameters and keep the most important ones, some parameters specific to
sub-layers, models are not offered at the top level of the given architecture by the ``Parametric API``, so if you need
to tweak those parameters missing from the main model, you can use ``LayerFlow API`` and create an instance of that
layer/model with desired parameters and pass it to the model.

Now, some of you mega smart minded ones, might be thinking, what if there was a way to pass instances of certain layers
but also just specify some parameter value for other layers and leave it to Teras to instantiate those layers, well I'm
just like you fr and you can do just that with the ``LayerFlow API``. Even though, the parameters specific to layers
aren't  included in the documentation of the ``LayerFlow`` version of those layers/models, but you can specify any
parameter that is offered by the ``Parametric API`` in the ``LayerFlow`` version of the model.

For instance, say we just want to pass an instance of our custom ``Categorical Embedding`` layer but instead of specify
an instance of the ``Encoder`` layer, we just want to modify the dropout rate for the ``FeedForward`` layer within it.
Well since we know that the ``TabTransformerClassifier``'s Parametric version exposes a ``feed_forward_dropout``
parameter, we can pass that keyword argument in the ``LayerFlow`` version of the ``TabTransformerClassifier``.

Here's how you'd do it in the code::

   from teras.layerflow.models import TabTransformerClassifier
   custom_categorical_embedding = CustomCategoricalFeatureEmbedding()
   model = TabTransformerClassifier(categorical_feature_emebdding=custom_categorical_embedding,
                                    feed_forward_dropout=0.5)


.. toctree::
   :maxdepth: 2
   :caption: Layers

   teras.layerflow.layers

.. toctree::
   :maxdepth: 2
   :caption: Models

   teras.layerflow.models
