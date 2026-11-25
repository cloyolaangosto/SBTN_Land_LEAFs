"""Tests for LS3 (Sun et al. 2024 LS-WPC / CSLE), src/sbtn_leaf/rusle_ls3.py."""
import math

import numpy as np
import pytest

pytest.importorskip("numba")
from sbtn_leaf.rusle_ls3 import csle_s, ls3_from_dem, ls3_m, segment_l  # noqa: E402


def _homs():
    """Himmelblau-Orlandini surface (Sun et al. 2024 Eq. 11), 0-50 m at 0.1 m, scaled x10: a 1 m grid."""
    x = np.arange(0, 50 + 1e-9, 0.1)
    X, Y = np.meshgrid(x, x[::-1])
    Z = 45 - 0.075 * ((((X - 25) / 5) ** 2 + (Y - 25) / 5 - 4) ** 2 + (((Y - 25) / 5) ** 2 + (X - 25) / 5 - 7) ** 2)
    return Z * 10


def test_homs_matches_paper_table1():
    ls, p = ls3_from_dem(_homs(), dx_m=1.0, dy_m=1.0, return_parts=True)
    slope = np.degrees(p["theta"])
    # Sun et al. 2024 Table 1, LS-WPC: slope 0.10-84.97 (mean 50.91), slope length max 407.65 (mean 64.96),
    # LS max 133.12 (mean 36.90)
    assert np.nanmax(slope) == pytest.approx(84.97, abs=0.05)
    assert np.nanmin(slope) == pytest.approx(0.10, abs=1e-9)
    assert np.nanmean(slope) == pytest.approx(50.91, abs=1.0)
    assert np.nanmean(p["lam_out"]) == pytest.approx(64.96, rel=0.10)
    assert np.nanmax(p["lam_out"]) == pytest.approx(407.65, rel=0.10)
    assert np.nanmean(ls) == pytest.approx(36.90, rel=0.10)
    assert np.nanmax(ls) == pytest.approx(133.12, rel=0.10)


def test_plane_accumulates_downslope():
    d, s = 10.0, 0.05                                    # 10 m cells, 5% slope falling to the east
    z = -s * d * np.tile(np.arange(40.0), (30, 1))
    ls, p = ls3_from_dem(z, dx_m=d, dy_m=d, channel_area_m2=1e12, return_parts=True)
    row = 15
    np.testing.assert_allclose(p["lam_out"][row, :-1], d * np.arange(1, 40))     # +1 cell per step east
    theta = math.atan(s)
    assert np.allclose(p["theta"][row, :-1], theta)
    lam_in, lam_out = p["lam_in"][row, 10], p["lam_out"][row, 10]
    m = 0.4                                              # 2.86 deg is in the 1.5-3 deg class
    L = (lam_out ** (m + 1) - lam_in ** (m + 1)) / ((lam_out - lam_in) * 22.13 ** m)
    assert ls[row, 10] == pytest.approx(L * (10.8 * math.sin(theta) + 0.03))


def test_slope_break_cuts_slope_length():
    d = 10.0
    prof = np.concatenate([-0.20 * d * np.arange(20), -0.20 * d * 19 - 0.02 * d * np.arange(1, 20)])
    z = np.tile(prof, (5, 1))                            # 20% slope, then 2%: decrease 90% > 0.5 -> cut
    _, p = ls3_from_dem(z, dx_m=d, dy_m=d, channel_area_m2=1e12, return_parts=True)
    # D8 slope points downstream, so column 19 (last steep cell) already has the 2% slope: the cut is 18 -> 19
    assert p["lam_in"][2, 19] == 0.0
    assert p["lam_out"][2, 18] == pytest.approx(19 * d)
    assert p["lam_in"][2, 25] > 0                        # the gentle part then accumulates again


def test_channel_restarts_slope_length():
    d = 10.0
    z = -0.05 * d * np.tile(np.arange(40.0), (3, 1))
    _, p = ls3_from_dem(z, dx_m=d, dy_m=d, channel_area_m2=10 * d * d, return_parts=True)
    assert np.all(p["lam_in"][1, 10:-1] == 0.0)          # area > 10 cells from column 10 on (1 row wide flow)


def test_s_continuous_and_m_classes():
    lo, hi = csle_s(np.radians([5.0 - 1e-6, 5.0 + 1e-6]))
    assert hi == pytest.approx(lo, abs=0.02)            # continuous at 5 deg with -0.50 (the paper prints -0.05)
    lo, hi = csle_s(np.radians([10.0 - 1e-6, 10.0 + 1e-6]))
    assert hi - lo == pytest.approx(0.43, abs=0.01)      # Liu et al. (1994) steep branch: known step at 10 deg
    assert list(ls3_m(np.radians([0.5, 1.0, 2.0, 3.0]))) == [0.2, 0.3, 0.4, 0.5]
    assert segment_l(np.array([0.0]), np.array([22.13]), np.array([0.5]))[0] == pytest.approx(1.0)


def test_geographic_grid_runs_and_masks_nan():
    z = np.add.outer(np.linspace(100, 0, 50), np.zeros(60))
    z[:5, :5] = np.nan
    ls = ls3_from_dem(z, lat_deg=45 - (np.arange(50) + 0.5) / 1200, res_deg=1 / 1200)
    assert np.isnan(ls[:5, :5]).all() and np.isfinite(ls[10:, 10:]).all()
