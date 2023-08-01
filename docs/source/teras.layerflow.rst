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



.. toctree::
   :maxdepth: 2
   :caption: Layers

   teras.layerflow.layers

.. toctree::
   :maxdepth: 2
   :caption: Models

   teras.layerflow.models