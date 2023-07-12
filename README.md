# Teras — A Unified Deep Learning Library for Tabualr Data

Teras, which is short for "Tabular Keras", aims to provide all the state of the art deep learning architectures (models/layers) for tabular data proposed by researchers. It inclues models ranging from Classificaiton and Regression to Data Generation (using GANs and VAEs) and Imputation. It also includes Preprocessing, Encoding and (Categorical and Numerical) Embedding layers. 

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

## Motivation
The main purposes of Teras are to:
1. Provide a uniform interface for all the different proposed architectures.
2. Further bridge the gap between research and application.
3. Be a one-stop for everything concerning deep learning for tabular data.
4. Accelerate research in tabular domain of deep learning by making it easier for researchers to access, use and experiment with exisiting architectures — saving them lots of valuable time.
