import pytest
import numpy as np
from mra_generate import *


def test_mean():
    arr = np.random.uniform(size=(100, 30))
    row_mean = np.mean(arr, axis=1)
    mean = np.mean(row_mean)

    assert pytest.approx(np.mean(arr)) == mean


def test_error_sanity():
    signal = np.array([1, 2, 3, 4, 5])
    assert pytest.approx(signal_observation_error(signal, signal)) == 0


@pytest.mark.parametrize('roll', [1, 2, 3, 4, 0])
def test_error_roll(roll):

    signal = np.array([1, 2, 3, 4, 5])

    assert pytest.approx(signal_observation_error(signal, np.roll(signal, roll))) == 0
