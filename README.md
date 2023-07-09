# Teras — A Unified Deep Learning Framework for Tabualr Data

Teras, which is short for "Tabular Keras", aims to provide all the state of the art deep learning architectures (models/layers) for tabular data proposed by researchers.

## Usage
Teras proviede two API for usage to satitate different levels of flexbility and accessbilit needs:
1. **Parametric API**: This is the default API, where user specifies values for parameters that are used in construction of any sub-layers or models within the architecture.
2. **LayerFlow API**: It maximizes flexbility and minimizes interface. Here, the user can pass any sub-layers or models instances as arguments to the given architecture (model/layer). It can be accessed through `teras.layerflow`


## Motivation
The main purposes of Teras are to:
1. Provide a uniform interface for all the different proposed architectures.
2. Further bridge the gap between research and application.
3. Be a one-stop for everything concerning deep learning for tabular data.
4. Accelerate research in tabular domain of deep learning by making it easier for researchers to access, use and experiment with exisiting architectures — saving them lots of valuable time.
