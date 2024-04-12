.. _getting-started:

Getting started with teras
==========================

teras strives to make tabular deep learning accessible.
It goes without saying, but since teras is based on Keras so all layers and
models available in teras are just wrappers around Keras layers and models
and hence provide seamless integration.


.. note::
   **teras v0.3 is now fully based on Keras 3, making everything available
   backend agnostic. It supports TensorFlow, JAX and PyTorch backends.**

.. warning::
    To use teras v0.3 you must have Keras 3 installed! It won't work with
    Keras 2.x

Installing teras
-----------------
You can install teras using pip as follows,

>>> pip install teras

Configuring backend
--------------------

You can export the environment variable ``KERAS_BACKEND`` or you can edit your
local config file at ``~/.keras/keras.json`` to configure your backend.
Available backend options are: "jax", "tensorflow", "torch".
Example:

>>> export KERAS_BACKEND="jax"

In Colab, you can do,

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "jax"
   import keras

For more Keras related configuration, please refer to
`Getting started with Keras <keras_getting_started>`_.


