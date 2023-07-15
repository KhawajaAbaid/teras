.. _getting_started:

Getting Started with Teras
=============================

In this tutorial, I'll give you an overview and everything you need to know to get started using Teras.

Introduction
--------------

Teras (short for Tabular Keras) is a unified deep learning library for Tabular Data that aims to be your one stop for
everything related to deep learning with tabular data.

It provides state of the art layers, models and architectures for all purposes, be it classification, regression,
pretraining or data generation and imputation using state of the art deep learning architectures.

While these state of the art architectures can be quite sophisticated, Teras — thanks to the incredible design of Keras —
abstracts away all the complications and sophistication and makes it easy as ever to access those models and put them
to use.

Not only that, everything available is highly customizable and modular, allowing for all variety of use cases.

Main Teras modules
-------------------------
Teras offers following main modules:

* ``teras.layerflow``: It is the LayerFlow API, offering maximum flexibility with minimal interface.
   It's an alternative to the default ``Parametric API``. You can read more about the difference between
   the two in the Teras APIs section below.
* ``teras.layers``: It contains all the layers for all of the architectures offered by Teras.
* ``teras.models``: It contains all of the models of all the architectures types, be it Classification, Regression,
   Pretraining etc. offered by Teras.
* ``teras.generative``: It contains state of the art models for Data Generation.
   (Currently it offers ``CTGAN`` and ``TVAE``).
* ``teras.impute``: It contains state of the art models for Data Imputation.
   (Currently it offers ``GAIN`` and ``PCGAIN``)
* ``teras.preprocessing``: It offers preprocessing classes for data transformation and data sampling that are
   required by highly sophisticated models specifically the data generation and imputation models.
* ``teras.ensemble``: It is a work in progress and aims to offers ensembling techniques making it easier than ever to
   ensemble your deep learning models, such as using Bagging or Stacking.
   (Currently it offers very basic version of these.)
* ``teras.utils``: It contains useful utility functions making life easier for Teras users
* ``teras.losses``: It contains custom losses for various architectures.

Teras APIs
---------------

Teras offers two different APIs for accessing and customizing the models to satiate different levels of accessibility
and flexibility needs.

Parametric API:
^^^^^^^^^^^^^^^^
It is the default API and something you're already familiar with — you import the model class and
specify the values for parameters that determine how the sub layers, models or blocks within the given model are
constructed.

For instance, specify the ``embedding_dim`` parameter during instantiation of ``TabTransformerClassifier`` and
that will be the dimensionality value used to construct the ``CategoricalFeatureEmbedding`` layer.
Simple enough, right?

LayerFlow API:
^^^^^^^^^^^^^^^^
It maximizes the flexibility while minimizing the interface. That may sound a bit contradictory at first, but let me
explain. Here, instead of specify individual parameters specific to sub layers/models, the user instead specifies
instances of those sub layers, models or blocks that are used in the given model architecture.

For instance, instead of specifying the embedding_dim parameter value, the user specifies an instance of
CategoricalFeatureEmbedding layer.

Now in this instance, we're just passing one parameter instead of another so it may not seem like much beneficial at
first glance but let me highlight how it can immensely help depending on your use case:

Since all you need to pass is an instance of layer, it can be any layer, there's no restriction that it must be
instance of ``CategoricalFeatureEmbedding`` layer — which means that you get complete control over not just customizing
the existing layers offered by ``Teras`` but also you can design/create an ``Embedding`` layer of your own that can
work in the place of the original ``CategoricalFeatureEmbedding`` layer or any other layer for that matter.
This is especially useful, if you're a designing a new architecture and want to rapidly test out new modifications of
the existing architectures by just plugging certain custom layers of your own in place of the default ones.
Pretty cool, right?

In many cases, to reduce the plethora of parameters and keep the most important ones, some parameters specific to
sub-layers, models are not offered at the top level of the given architecture by the ``Parametric API``,
so if you need to tweak those parameters missing from the main model, you can use ``LayerFlow API`` and
create an instance of that layer/model with desired parameters and pass it to the model.

Now, some of you mega smart minded ones, might be thinking, what if there was a way to pass instances of certain layers
but also just specify some parameter value for other layers and leave it to Teras to instantiate those layers, well I'm
just like you fr and you can do just that with the ``LayerFlow API``.
Even though, the parameters specific to layers aren't included in the documentation of the ``LayerFlow`` version of
those layers/models, but you can specify any parameter that
is offered by the ``Parametric API`` in the ``LayerFlow API`` version of the model.

For instance, say we just want to pass an instance of our custom ``CategoricalEmbedding`` layer but instead of specify
an instance of the ``Encoder`` layer, we just want to modify the dropout rate for the ``FeedForward`` layer within it.
Well since we know that the ``TabTransformerClassifier``'s Parametric version exposes a ``feed_forward_dropout``
parameter, we can pass that keyword argument in the LayerFlow version of the ``TabTransformerClassifier``.

Here's how you'd do it in the code::

   from teras.layerflow.models import TabTransformerClassifier
   custom_categorical_embedding = CustomCategoricalFeatureEmbedding()
   model = TabTransformerClassifier(categorical_feature_emebdding=custom_categorical_embedding,
                                feed_forward_dropout=0.5)

Wrapping it up!
-------------------
That's pretty much all you need to know to get started with Teras.

You can find the tutorials for classification, pretraining, data imputation and data generation using Teras in the
`tutorials directory <https://github.com/KhawajaAbaid/teras/tree/main/tutorials>`_ of
Teras's `github repository <https://github.com/KhawajaAbaid/teras>`_.

And that wraps up our getting started guide.

If you need more help, consult documentation, and other available resources and if that still leaves you with
questions, feel free to raise an issue or email me khawaja.abaid@gmail.com

If you find Teras useful, please consider giving it a star on `GitHub <https://github.com/KhawajaAbaid/teras>`_
and sharing it with others!

Thank you!