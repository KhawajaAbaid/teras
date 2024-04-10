``teras.models`` module
=======================

.. py:currentmodule:: teras.models

.. automodule:: teras.models


Backbones
----------

.. autosummary::
   :toctree: _autosummary

    FTTransformerBackbone
    SAINTBackbone
    TabNetEncoderBackbone
    TabTransformerBackbone
    TransformerEncoderBackbone

Pretrainers
------------

.. autosummary::
   :toctree: _autosummary

    SAINTPretrainer
    TabNetPretrainer
    TabNetDecoder
    TabTransformerRTDPretrainer
    TabTransformerMLMPretrainer

Generative
------------

.. autosummary::
   :toctree: _autosummary

    CTGAN
    CTGANGenerator
    CTGANDiscriminator
    GAIN
    GAINGenerator
    GAINDiscriminator
    PCGAIN
    TVAE
    TVAEEncoder
    TVAEDecoder

Task Models
------------

.. autosummary::
   :toctree: _autosummary

    Classifier
    Regressor
