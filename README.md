# Teras — A Unified Deep Learning Library for Tabualr Data

Teras (short for Tabular Keras) is a unified deep learning library for Tabular Data that aims to be your one stop for everything related to deep learing with tabular data.

It provides state of the art layers, models and arhitectures for all purposes, be it classificaiton, regression or even data generation and imputation using state of the art deep learning architectures. 

It also includes Preprocessing, Encoding and (Categorical and Numerical) Embedding layers. 

While these state of the art architectures can be quite sophisticated, Teras, thanks to the increidble design of Keras, abstracts away all the complications and sophistication and makes it easy as ever to access those models and put them to use.

Not only that, everything available is highly customizable and modular, allowing for all variety of use cases.

## Getting Started
Read our [Getting Started Guide](https://github.com/KhawajaAbaid/teras/blob/main/tutorials/getting_started.ipynb) to...*drum roll* get started with Teras.

## Usage
Teras provieds two API for usage to satitate different levels of flexbility and accessbility needs:
1. **Parametric API**: This is the default API, where user specifies values for parameters that are used in construction of any sub-layers or models within the architecture.
```
from teras.models import TabNetClassifier

model = TabNetClassifier(num_classes=2, features_metadata=features_metadata)
```
2. **LayerFlow API**: It maximizes flexbility and minimizes interface. Here, the user can pass any sub-layers or models instances as arguments to the given architecture (model/layer). It can be accessed through `teras.layerflow`
```
from teras.layerflow.models import TabNetClassifier
from teras.layerflow.layers import TabNetEncoder, TabNetClassificationHead

encoder = TabNetEncoder()
head = TabNetClassificationHead(num_classes=2)
model = TabNetClassifier(features_metadata=features_metadata,
                         encoder=encoder,
                         head=head)
```
You can read more about the difference between the two in the Teras APIs section in the [Getting Started Guide](https://github.com/KhawajaAbaid/teras/blob/main/tutorials/getting_started.ipynb).

## Main Teras Modules
Teras offers following main modules:

1. `teras.layerflow`: It is the LayerFlow API, offering maximum flebility with minimal interface. It's an alternative to the default Parametric API. You can read more about the difference between the two in the Teras APIs section in the [Getting Started Guide](https://github.com/KhawajaAbaid/teras/blob/main/tutorials/getting_started.ipynb).
2. `teras.layers`: It contains all the layers for all of the architectures offered by Teras.
3. `teras.models`: It contains all of the models of all the architectures types, be it Classificaiton, Regresssion etc offered by Teras.
4. `teras.generative`: It contains state of the art models for Data Generation. (Currently it offers `CTGAN` and `TVAE`).
5. `teras.impute`: It contains state of the art models for Data Imputation. (Currently it offers `GAIN` and `PCGAIN`)
6. `teras.preprocessing`: It offers preprocessing classes for data transformation and data sampling that are required by highly sophisticated models specifically the data generation and imputation models.
7. `teras.ensemble`: It is a work in progress and aims to offers ensembling techniques making it easier than ever to ensemble your deep learning models, such as using Bagging or Stacking. (Currently it offers very basic version of these.)
8. `teras.utils`: It contains useful utility functions making life easier for Teras users
9. `teras.losses`: It contains custom losses for various architectures.

## Motivation
The main purposes of Teras are to:
1. Provide a uniform interface for all the different proposed architectures.
2. Further bridge the gap between research and application.
3. Be a one-stop for everything concerning deep learning for tabular data.
4. Accelerate research in tabular domain of deep learning by making it easier for researchers to access, use and experiment with exisiting architectures — saving them lots of valuable time.


## Support
If you find Teras useful, consider supporting the project. I've been working on this for the past ~3 months full time and plan to continue to do so. I also have many future plans for it but my currently laptop is quite old which makes it impossible for me to test highly demanding workflows let alone rapidly test and iterate. So your support will be very vital in the betterment of this project.
Thank you!

[![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/KhawajaAbaid)
[![Patreon](https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/KhawajaAbaid)
