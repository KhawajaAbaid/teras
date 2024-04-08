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

import types
import keras



# Copied from Keras-CV
# https://github.com/keras-team/keras-cv/blob/master/keras_cv/api_export.py
def maybe_register_serializable(symbol, package):
    if isinstance(symbol, types.FunctionType) or hasattr(symbol, "get_config"):
        keras.saving.register_keras_serializable(package=package)(symbol)


# if namex:
#     class teras_export(namex.export):
#         def __init__(self, path, package="teras"):
#             super().__init__(package="teras", path=path)
#             self.package = package
#
#         def __call__(self, symbol):
#             maybe_register_serializable(symbol, self.package)
#             return super().__call__(symbol)
#
# else:
class teras_export:
    def __init__(self, path, package="teras"):
        self.package = package

    def __call__(self, symbol):
        maybe_register_serializable(symbol, self.package)
        return symbol
