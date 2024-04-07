import pytest
from keras.backend import backend


skip_on_torch = pytest.mark.skipif(
    backend() == "torch", reason="Test enters infinite loop on `torch` backend."
)
