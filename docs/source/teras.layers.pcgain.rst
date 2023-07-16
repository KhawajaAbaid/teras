PCGAIN Layers
==========================

.. admonition:: Note!

   **PCGAIN** architecture is based heavily on the **GAIN** architecture and uses the same ``GeneratorBlock`` and
   ``DiscriminatorBlock`` layers as **GAIN**, hence  **PCGAIN** doesn't implement those layers. But you can still access
   these layers from ``teras.layers.pcgain`` which just contains a reference to layers in ``teras.layers.gain`` for the
   sake of convenience.

Here are quick links to the documentation of the **GAIN** layers used by **PCGAIN**.

.. toctree::
   :maxdepth: 2

   teras.layers.gain
