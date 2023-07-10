from tensorflow import keras
from tensorflow.keras import layers, models
from teras.models.tvae import (Encoder as _BaseEncoder,
                               Decoder as _BaseDecoder,
                               TVAE as _BaseTVAE)
from typing import Union

LAYER_OR_MODEL = Union[layers.Layer, models.Model]


class Encoder(_BaseEncoder):
    """
    Encoder for the TVAE model with LayerFlow design.
    TVAE is proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        compression_block: `layers.Layer`, `List[layers.Layer]` or `models.Model`,
            An instance of Keras Dense layer, list of layers or Keras model
            that can work as the compression block for the Encoder model.
            If None, a compression made up of two dense layers of 128 dimensionality
            is used.
        latent_dim: `int`, default 128.
            Dimensionality of the learned latent space
    """
    def __init__(self,
                 compression_block: LAYER_OR_MODEL = None,
                 latent_dim: int = 128,
                 **kwargs):
        super().__init__(latent_dim=latent_dim,
                         **kwargs)

        if compression_block is not None:
            if isinstance(compression_block, (layers.Layer, models.Model)):
                # leave it as is
                compression_block = compression_block
            elif isinstance(compression_block, (list, tuple)):
                compression_block = models.Sequential(compression_block,
                                                      name="encoder_compression_block")
            else:
                raise TypeError("`compression_block` can either be a Keras layer, list of layers or a keras model "
                                f"but received type: {type(compression_block)} which is not supported.")
            self.compression_block = compression_block

    def get_config(self):
        config = super().get_config()
        new_config = {'compression_block': keras.layers.serialize(self.compression_block),
                      }
        config.update(new_config)
        return config


class Decoder(_BaseDecoder):
    """
    Encoder for the TVAE model as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        data_dim: `int`,
            Dimensionality of the input dataset.
        decompression_block: `layers.Layer` or `models.Model`, decompresses the compressed
            latent dimension data back to original data dimensions.
            Note, the last layer of decompression_block must project the data back to
            original dimensions i.e. `data_dim`.
    """
    def __init__(self,
                 data_dim: int = None,
                 decompression_block: LAYER_OR_MODEL = None,
                 **kwargs):
        super().__init__(data_dim=data_dim,
                         **kwargs)

        if decompression_block is not None:
            if isinstance(decompression_block, (layers.Layer, models.Model)):
                # leave it as is
                decompression_block = decompression_block
            elif isinstance(decompression_block, (list, tuple)):
                decompression_block = models.Sequential(decompression_block,
                                                        name="decoder_decompression_block")
            else:
                raise TypeError("`decompression_block` can either be a Keras layer, list of layers or a keras model "
                                f"but received type: {type(decompression_block)} which is not supported.")
            self.decompression_block = decompression_block

    def get_config(self):
        config = super().get_config()
        new_config = {'decompression_block': keras.layers.serialize(self.decompression_block),
                      }
        config.update(new_config)
        return config


class TVAE(_BaseTVAE):
    """
    TVAE model with LayerFlow design.
    TVAE is a tabular data generation architecture proposed by the
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        data_dim: `int`, dimensionality of the input dataset.
            It should be the dimensionality of the dataset that is passed to the `fit`
            method and not necessarily the dimensionality of the original input data
            as sometimes data transformations can alter the dimensionality of the data.
            Conveniently, if you're using DataSampler class, you can access `.data_dim`
            attribute to get the dimensionality of data.
        meta_data: `dict`,
            A dictionary containing metadata for all features in the input data.
            This metadata is computed during the data transformation step and can be accessed
            from `.get_meta_data()` method of DataTransformer instance.
        encoder: `layers.Layer` or `models.Model`.
            An instance of `TVAEEncoder` or any custom keras layer/model that can work
            in its place.
            It encodes or compresses data to useful hidden dimensional representations.
            If `None`, a default encoder is constructed with (128, 128)
            dimensions is constructed.
            You can import the `TVAEEncoder` as follows,
                >>> from teras.layerflow.models import TVAEEncoder
        decoder: `layers.Layer` or `models.Model`
            An instance of `TVAEDecoder` or any custom keras model/layer that can work
            in its palce.
            It decodes or decompresses from latent dimensions to data dimensions.
            If `None`, a default decoder is constructed with (128, 128) hidden
            dimensions along with a dense output layer that projects the data
            back to original dimensions.
            You can import the `TVAEDecoder` as follows,
                >>> from teras.layerflow.models import TVAEDecoder
        latent_dim: `int`, default 128,
            Dimensionality of the learned latent space
        loss_factor: `float`, default 2,
            Hyperparameter used in the computation of ELBO loss for TVAE.
            It controls how much the cross entropy loss contributes to the overall loss.
            It is directly proportional to the cross entropy loss.
    """
    def __init__(self,
                 data_dim: int = None,
                 meta_data: dict = None,
                 encoder: LAYER_OR_MODEL = None,
                 decoder: LAYER_OR_MODEL = None,
                 latent_dim: int = 128,
                 loss_factor=2,
                 **kwargs):
        super().__init__(data_dim=data_dim,
                         meta_data=meta_data,
                         latent_dim=latent_dim,
                         loss_factor=loss_factor,
                         **kwargs)

        if encoder is not None:
            self.encoder = encoder

        if decoder is not None:
            self.decoder = decoder

    def get_config(self):
        config = super().get_config()
        new_config = {'encoder': keras.layers.serialize(self.encoder),
                      'decoder': keras.layers.serialize(self.decoder)
                      }
        config.update(new_config)
        return config
