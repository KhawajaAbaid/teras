Common Layers
=====================================

Teras ``layerflow.common`` module provides `LayerFlow` version of common layers used across architectures.
Sometimes these layers are subclassed to implement the architecture specific variant of these layers,
for instance, ``ClassificationHead`` is subclassed to build ``SAINTClassificationHead``, while other times they are
used as is, for instance, the common transformer layers are used as is in the ``TabTransformer`` architecture.

.. toctree::
   :maxdepth: 2

   teras.layerflow.layers.common.transformer
