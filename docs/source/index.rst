teras - a unified deep learning library for tabular data!
=========================================================

Hello, world! Welcome to the documentation of teras.

teras (short for Tabular Keras) goal is to be your one stop for everything
related to deep learning with tabular data and to accelerate tabular research.

.. note::
   **teras v0.3 is now fully based on Keras 3, making everything available
   backend agnostic. It supports TensorFlow, JAX and PyTorch backends.**

teras provides state of the art layers, models and architectures for all
purposes, be it classification, regression or even data generation and
imputation using state of the art deep learning architectures.

It also includes functions and classes for preprocessing data for complex
architectures, making it extremely simple to transform your data in the
expected format, saving you loads of hassle and time!

While these state of the art architectures can be quite sophisticated, Teras,
thanks to the incredible design of Keras, abstracts away all the complications
and sophistication and makes it easy as ever to access those models and put them
to use.

Not only that, everything available is highly customizable and modular, allowing
for all variety of use cases.


.. grid:: 3
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Classification/Regression Models
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Teras offers state of the art architectures as backbones for building
      customizable and modular models for classification and regression quickly
      and easily!

   .. grid-item-card:: Generative Models
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Generate synthetic dataset based on a small dataset or impute a dataset
      with missing features using the state of the art generative models offers
      by teras.


   .. grid-item-card:: Preprocessing Classes
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      Setting up your data to be fed into the specialized architectures can
      be a challenge, but teras makes it super easy and intuitive by offering
      DataTransformer and DataSampler classes.

.. grid:: 3

    .. grid-item-card:: :material-regular:`rocket_launch;2em` Getting Started
      :columns: 12 6 6 4
      :link: getting-started
      :link-type: ref
      :class-card: getting-started

    .. grid-item-card:: :material-regular:`library_books;2em` User Guides
      :columns: 12 6 6 4
      :link: user-guide
      :link-type: ref
      :class-card: user-guide

    .. grid-item-card:: :material-regular:`laptop_chromebook;2em` Developer Docs
      :columns: 12 6 6 4
      :link: developer-guide
      :link-type: ref
      :class-card: developer-docs


Installation
-------------

>>> pip install teras


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   getting_started
   notebooks/backbones
   notebooks/data_imputation


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Further Resources

   teras
