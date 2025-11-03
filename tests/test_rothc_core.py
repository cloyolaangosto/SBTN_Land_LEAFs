import warnings

import numpy as np

from sbtn_leaf.RothC_Core import RMF_Tmp


def test_rmf_tmp_avoids_warnings_near_lower_bound():
    temps = np.array([
        -25.0,
        -18.27,
        -10.0,
        -5.0,
        -4.999,
        0.0,
        10.0,
    ])

    with np.errstate(over="raise"):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = RMF_Tmp(temps)

    assert record == []

    expected = np.zeros_like(temps, dtype=float)
    mask = temps >= -5.0
    expected[mask] = 47.91 / (np.exp(106.06 / (temps[mask] + 18.27)) + 1.0)

    np.testing.assert_allclose(result, expected)


def test_rmf_tmp_scalar_behaviour():
    with np.errstate(over="raise"):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            value = RMF_Tmp(-20.0)

    assert record == []
    assert value == 0.0
