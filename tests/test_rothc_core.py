import warnings

import numpy as np

from sbtn_leaf.RothC_Core import RMF_Tmp, RMF_TRM, _partition_to_bio_hum


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

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = RMF_Tmp(temps)

    assert record == []

    expected = np.zeros_like(temps, dtype=float)
    mask = temps >= -5.0
    expected[mask] = 47.91 / (np.exp(106.06 / (temps[mask] + 18.27)) + 1.0)

    np.testing.assert_allclose(result, expected)


def test_rmf_tmp_scalar_behaviour():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        value = RMF_Tmp(-20.0)

    assert record == []
    assert value == 0.0


def test_partition_to_bio_hum_scalar_and_array_consistency():
    clay_value = 30.0
    x_scalar = 1.67 * (1.85 + 1.60 * np.exp(-0.0786 * clay_value))
    carbon_scalar = 5.0

    bio_scalar, hum_scalar = _partition_to_bio_hum(x_scalar, carbon_scalar)

    expected_bio_scalar = carbon_scalar * (0.46 / (x_scalar + 1.0))
    expected_hum_scalar = carbon_scalar * (0.54 / (x_scalar + 1.0))

    assert isinstance(bio_scalar, float)
    assert isinstance(hum_scalar, float)
    np.testing.assert_allclose(bio_scalar, expected_bio_scalar)
    np.testing.assert_allclose(hum_scalar, expected_hum_scalar)

    x_array = np.array([x_scalar, x_scalar + 0.5, x_scalar + 1.0])
    carbon_array = np.array([0.5, 1.5, 2.5])
    bio_array, hum_array = _partition_to_bio_hum(x_array, carbon_array)

    expected_bio_array = carbon_array * (0.46 / (x_array + 1.0))
    expected_hum_array = carbon_array * (0.54 / (x_array + 1.0))

    np.testing.assert_allclose(bio_array, expected_bio_array)
    np.testing.assert_allclose(hum_array, expected_hum_array)

    # Ensure consistent behaviour between scalar and array pathways
    array_scalar_bio, array_scalar_hum = _partition_to_bio_hum(
        np.array([x_scalar]), np.array([carbon_scalar])
    )

    np.testing.assert_allclose(array_scalar_bio[0], bio_scalar)
    np.testing.assert_allclose(array_scalar_hum[0], hum_scalar)


def test_rmf_trm_selects_all_nodes():
    sand = np.array([40.0, 40.0, 36.0, 30.0])
    soc = np.array([80.0, 70.0, 50.0, 50.0])

    trm_dpm, trm_rpm, trm_bio, trm_hum = RMF_TRM(sand, soc)

    expected_dpm = np.array([1.54, 1.71, 1.54, 0.72])
    expected_rpm = np.array([0.35, 0.35, 2.15, 0.97])
    expected_bio = np.array([1.42, 0.38, 2.38, 0.99])
    expected_hum = np.array([0.42, 0.87, 2.93, 0.94])

    np.testing.assert_allclose(trm_dpm, expected_dpm)
    np.testing.assert_allclose(trm_rpm, expected_rpm)
    np.testing.assert_allclose(trm_bio, expected_bio)
    np.testing.assert_allclose(trm_hum, expected_hum)
