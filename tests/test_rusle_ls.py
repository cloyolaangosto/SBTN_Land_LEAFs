"""Tests for the MERIT-based LS factor in sbtn_leaf.rusle_ls (synthetic tiles and archives)."""

import math
import tarfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from sbtn_leaf import rusle_ls as ls

RES = 1.0 / 1200.0        # MERIT 3"
TILE = 200                # small synthetic tiles (real MERIT tiles are 6000 px)


def _pad(z):
    p = np.full((z.shape[0] + 2, z.shape[1] + 2), np.nan)
    p[1:-1, 1:-1] = z
    return p


def _reference_ls(sin_t, A, D, x=1.0):
    beta = (sin_t / 0.0896) / (3 * sin_t ** 0.8 + 0.56)
    m = beta / (1 + beta)
    L = ((A + D ** 2) ** (m + 1) - A ** (m + 1)) / (D ** (m + 2) * x ** m * 22.13 ** m)
    tan_t = sin_t / math.sqrt(1 - sin_t ** 2)
    S = 10.8 * sin_t + 0.03 if tan_t < 0.09 else 16.8 * sin_t - 0.5
    return L * S


def test_flat_ground_gives_unit_L_and_minimum_S():
    z = np.full((4, 5), 100.0)
    out = ls.ls_factor_desmet_govers(_pad(z), np.full((4, 5), 0.05), np.zeros(4), RES)
    np.testing.assert_allclose(out, 0.03)                      # m = 0 -> L = 1; S = 0.03


@pytest.mark.parametrize("tan_t", [0.05, 0.2])
def test_north_facing_ramp_matches_formula(tan_t):
    dy = ls.EARTH_RADIUS_M * math.radians(RES)
    rows = np.arange(6)
    z = np.repeat((-rows * dy * tan_t)[:, None], 4, axis=1)   # descends to the south
    lat = np.zeros(6)
    upa = np.full((6, 4), 1.0)                                  # km2, capped by lambda_max
    out = ls.ls_factor_desmet_govers(_pad(z), upa, lat, RES, lambda_max=300)

    D = math.sqrt(dy * dy)                                      # at the equator dx == dy
    sin_t = tan_t / math.sqrt(1 + tan_t ** 2)
    np.testing.assert_allclose(out, _reference_ls(sin_t, 300 * D, D), rtol=1e-9)


def test_lambda_cap_and_channel_mask():
    dy = ls.EARTH_RADIUS_M * math.radians(RES)
    z = np.repeat((-np.arange(3) * dy * 0.1)[:, None], 3, axis=1)
    upa = np.array([[0.01, 0.5, 50.0]] * 3)                     # A = 1e4 - D^2 ~ 1.4e3 m2 < cap of 100 * D ~ 9.3e3
    capped = ls.ls_factor_desmet_govers(_pad(z), upa, np.zeros(3), RES, lambda_max=100)
    np.testing.assert_allclose(capped[:, 1], capped[:, 2])      # both above the cap -> same LS
    assert np.all(capped[:, 0] < capped[:, 1])
    masked = ls.ls_factor_desmet_govers(_pad(z), upa, np.zeros(3), RES, lambda_max=100, channel_upa_km2=10)
    assert np.isnan(masked[:, 2]).all() and np.isfinite(masked[:, :2]).all()


def test_missing_neighbours_fall_back_to_one_sided():
    z = np.array([[0.0, 1.0, 4.0, 9.0]])                        # quadratic along x
    p = _pad(z)
    dzdx, _ = ls.slope_components(p, 1.0, np.ones(1))
    np.testing.assert_allclose(dzdx[0], [1.0, 2.0, 4.0, 5.0])  # edges one-sided, interior central


def test_aggregate_to_5min_sums_and_counts():
    a = np.arange(16, dtype=float).reshape(4, 4)
    a[0, 0] = np.nan
    s, n = ls.aggregate_to_5min(a, factor=2)
    assert n[0, 0] == 3 and s[0, 0] == 1 + 4 + 5
    assert n[1, 1] == 4 and s[1, 1] == 10 + 11 + 14 + 15
    with pytest.raises(ValueError):
        ls.aggregate_to_5min(a[:3], factor=2)


# --- synthetic archives ------------------------------------------------------------------------------
def _surface(lon, lat):
    """Smooth hilly surface (m)."""
    return 300 + 200 * np.sin(np.radians(lon) * 400) * np.cos(np.radians(lat) * 300)


def _write_tile(fp, west, north, kind):
    cols = west + (np.arange(TILE) + 0.5) * RES
    rows = north - (np.arange(TILE) + 0.5) * RES
    lon, lat = np.meshgrid(cols, rows)
    data = _surface(lon, lat) if kind == "dem" else np.full_like(lon, 0.1)
    data[0, 0] = -9999                                          # one nodata pixel
    with rasterio.open(fp, "w", driver="GTiff", height=TILE, width=TILE, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(west, north, RES, RES), nodata=-9999) as dst:
        dst.write(data.astype(np.float32), 1)
    return lon, lat, data


def _make_archives(tmp_path):
    """2 x 2 tiles of TILE px, as MERIT-style dem and upa tars, in a folder with a comma and spaces."""
    folder = tmp_path / "World Wildlife Fund, Inc" / "merit"
    folder.mkdir(parents=True)
    size = TILE * RES
    tars = {}
    for kind in ("dem", "upa"):
        tar_fp = folder / f"{kind}_test.tar"
        with tarfile.open(tar_fp, "w") as tf:
            for i, north in enumerate((10 + 2 * size, 10 + size)):
                for j, west in enumerate((20.0, 20 + size)):
                    fp = folder / f"t{i}{j}_{kind}.tif"
                    _write_tile(fp, west, north, kind)
                    tf.add(fp, arcname=f"{kind}_test/{fp.name}")
        tars[kind] = tar_fp
    return tars, size


def _full_mosaic_ls(size):
    """LS computed on one array covering all 4 tiles (the stitching reference)."""
    n = 2 * TILE
    cols = 20 + (np.arange(n) + 0.5) * RES
    rows = 10 + 2 * size - (np.arange(n) + 0.5) * RES
    lon, lat = np.meshgrid(cols, rows)
    z = _surface(lon, lat).astype(np.float32).astype(np.float64)
    upa = np.full_like(z, 0.1, dtype=np.float32).astype(np.float64)
    for r in (0, TILE):                                         # the nodata corner pixel of each tile
        for c in (0, TILE):
            z[r, c] = np.nan
            upa[r, c] = np.nan
    return ls.ls_factor_desmet_govers(_pad(z), upa, rows, RES)


def test_process_archive_stitches_tiles_exactly(tmp_path):
    tars, size = _make_archives(tmp_path)                      # 200 px tiles = 2 x 2 cells of 5' (100 px)
    out_fp = tmp_path / "parts" / "ls_sums_test.tif"
    sums, transform = ls.process_archive(tars["dem"], tars["upa"], out_fp, block_rows=100, show_progress=False)

    ref = _full_mosaic_ls(size)
    ref_sums = ls.aggregate_to_5min(ref, 100)
    assert sums.shape[0] == len(ls.LS_BANDS) == 2
    np.testing.assert_allclose(sums[0], ref_sums[0], rtol=1e-10)
    np.testing.assert_array_equal(sums[1], ref_sums[1])
    assert transform.c == pytest.approx(20.0) and transform.f == pytest.approx(10 + 2 * size)
    with rasterio.open(out_fp) as src:
        assert src.count == 2 and src.descriptions == ls.LS_BANDS

    # without neighbours only the cells touching internal tile edges change
    one_sided, _ = ls.process_archive(tars["dem"], tars["upa"], None, block_rows=100,
                                      use_neighbours=False, show_progress=False)
    changed = ~np.isclose(one_sided[0], sums[0], rtol=1e-12)
    touches_inner_edge = np.zeros_like(changed)
    touches_inner_edge[[1, 2], :] = True                        # cells either side of the inner tile lines
    touches_inner_edge[:, [1, 2]] = True
    assert changed[touches_inner_edge].any()
    assert not changed[~touches_inner_edge].any()


def test_mosaic_places_parts_and_divides(tmp_path):
    parts = tmp_path / "parts"
    sums = np.zeros((2, 2, 3))
    sums[0], sums[1] = 6.0, 2.0
    ls.write_sums(parts / "ls_sums_a.tif", sums, from_origin(0.0, 0.0, ls.FIVE_MIN_DEG, ls.FIVE_MIN_DEG))
    out = ls.mosaic_ls(parts, tmp_path / "ls.tif")
    r0, c0 = round(90 / ls.FIVE_MIN_DEG), round(180 / ls.FIVE_MIN_DEG)
    np.testing.assert_allclose(out[r0:r0 + 2, c0:c0 + 3], 3.0)
    assert np.isnan(out[0, 0]) and np.isfinite(out).sum() == 6
    with rasterio.open(tmp_path / "ls.tif") as src:
        np.testing.assert_array_equal(src.read(1), out)


def test_mosaic_rejects_old_four_band_parts(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    with rasterio.open(parts / "ls_sums_old.tif", "w", driver="GTiff", height=1, width=1, count=4,
                       dtype="float64", crs="EPSG:4326", transform=from_origin(10, 10, 1, 1)) as dst:
        dst.write(np.ones((4, 1, 1)))
    with pytest.raises(ValueError, match="bands"):
        ls.mosaic_ls(parts, tmp_path / "ls.tif")


def test_merit_box_names():
    names = ls.merit_box_names(lat_bands=(30, -30), lon_start=-30, lon_end=30)
    assert names == ["n30w030", "n30e000", "s30w030", "s30e000"]
