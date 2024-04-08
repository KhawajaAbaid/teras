# Copyright 2024 Khawaja Abaid Ullah
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Backbones
from teras._src.models.backbones.transformer.encoder import (
    TransformerEncoderBackbone as TransformerEncoderBackbone)
from teras._src.models.backbones.ft_transformer.ft_transformer import (
    FTTransformerBackbone as FTTransformerBackbone)
from teras._src.models.backbones.tab_transformer.tab_transformer import (
    TabTransformerBackbone as TabTransformerBackbone)
from teras._src.models.backbones.saint.saint import (
    SAINTBackbone as SAINTBackbone)
from teras._src.models.backbones.tabnet.encoder import (
    TabNetEncoderBackbone as TabNetEncoderBackbone)


# Pretrainers
from teras._src.models.pretrainers.tab_transformer import (
    TabTransformerRTDPretrainer as TabTransformerRTDPretrainer,
    TabTransformerMLMPretrainer as TabTransformerMLMPretrainer
)
from teras._src.models.pretrainers.saint import (
    SAINTPretrainer as SAINTPretrainer)
from teras._src.models.pretrainers.tabnet.tabnet import (
    TabNetPretrainer as TabNetPretrainer)
from teras._src.models.pretrainers.tabnet.decoder import (
    TabNetDecoder as TabNetDecoder
)


# GANs
from teras._src.models.gans.gain.gain import GAIN as GAIN
from teras._src.models.gans.gain.generator import GAINGenerator as GAINGenerator
from teras._src.models.gans.gain.discriminator import (
    GAINDiscriminator as GAINDiscriminator)
from teras._src.models.gans.pcgain.pcgain import PCGAIN as PCGAIN
from teras._src.models.gans.ctgan.ctgan import CTGAN as CTGAN
from teras._src.models.gans.ctgan.generator import (
    CTGANGenerator as CTGANGenerator)
from teras._src.models.gans.ctgan.discriminator import (
    CTGANDiscriminator as CTGANDiscriminator)


# Autoencoders
from teras._src.models.autoencoders.tvae.tvae import TVAE as TVAE
from teras._src.models.autoencoders.tvae.encoder import (
    TVAEEncoder as TVAEEncoder)
from teras._src.models.autoencoders.tvae.decoder import (
    TVAEDecoder as TVAEDecoder)


# Task models
from teras._src.models.tasks.regression import Regressor as Regressor
from teras._src.models.tasks.classification import Classifier as Classifier
