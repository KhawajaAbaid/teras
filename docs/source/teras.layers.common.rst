Common Layers
===========================

Teras ``common`` module provides common layers used across architectures. Sometimes these layers are subclassed to
implement the architecture specific variant of these layers, for instance, ``ClassificationHead`` is subclassed to build
``SAINTClassificationHead``, while other times they are used as is, for instance, the common transformer layers are used
as is in the ``TabTransformer`` architecture.

.. toctree::
   :maxdepth: 2
   :caption: Head

   teras.layers.common.head



.. toctree::
   :maxdepth: 2
   :caption: Transformer

   teras.layers.common.transformer
