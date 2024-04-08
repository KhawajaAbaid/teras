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
# limitatiFions under the License.

from keras.backend import backend

if backend() == "jax":
    from teras._src.backend.jax.trainers import *
else:
    # raise ImportError(
    #     "`teras.trainers` module is only available for `jax` backend. "
    #     "Please set your `KERAS_BACKEND` environment variable to `jax` before "
    #     "importing `teras.trainers`.")
    pass