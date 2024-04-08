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


# CTGAN
from teras._src.losses.ctgan import (
    ctgan_generator_loss as ctgan_generator_loss,
    ctgan_discriminator_loss as ctgan_discriminator_loss)

# SAINT
from teras._src.losses.saint import (
    saint_denoising_loss as saint_denoising_loss,
    saint_constrastive_loss as saint_constrastive_loss)

# TabNet
from teras._src.losses.tabnet import (
    tabnet_reconstruction_loss as tabnet_reconstruction_loss)
