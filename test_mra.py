import pytest
import numpy as np


def test_mean():
    arr = np.random.uniform(size=(100, 30))
    row_mean = np.mean(arr, axis=1)
    mean = np.mean(row_mean)

    assert pytest.approx(np.mean(arr), mean)
