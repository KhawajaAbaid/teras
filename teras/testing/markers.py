import pytest
from keras.backend import backend


skip_save_and_load_on_torch = pytest.mark.skipif(
    backend() == "torch", reason="Saving and loading enters infinite loop on "
                                 "`torch` backend."
)
