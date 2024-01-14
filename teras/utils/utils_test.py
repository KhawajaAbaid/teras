import os
from teras.utils.utils import get_tmp_dir


def test_get_tmp_dir():
    assert os.path.exists(get_tmp_dir()) == True
