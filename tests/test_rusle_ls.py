"""Tests for the MERIT-based LS factor in sbtn_leaf.rusle_ls (synthetic tiles and archives)."""

import json
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


def test_slope_clamp():
    dy = ls.EARTH_RADIUS_M * math.radians(RES)
    D = dy                                                      # equator: dx == dy
    rows = np.arange(5)
    upa = np.full((5, 4), 1.0)                                  # km2, capped by lambda_max

    def ramp(tan_t, **kw):
        z = np.repeat((-rows * dy * tan_t)[:, None], 4, axis=1)   # descends to the south, x = 1
        return ls.ls_factor_desmet_govers(_pad(z), upa, np.zeros(5), RES, lambda_max=122, **kw)

    sin_cap = math.sin(math.radians(26.6))
    np.testing.assert_allclose(ramp(1.0), _reference_ls(sin_cap, 122 * D, D), rtol=1e-9)        # 45 deg -> 26.6
    np.testing.assert_allclose(ramp(1.0, max_slope_deg=None),
                               _reference_ls(math.sin(math.pi / 4), 122 * D, D), rtol=1e-9)    # no clamp
    np.testing.assert_allclose(ramp(0.3), ramp(0.3, max_slope_deg=None), rtol=1e-12)        # below the clamp

    # the clamp changes the slope, not the aspect term: a diagonal 45-deg slope keeps x = sqrt(2)
    cols = np.arange(4)
    step = dy / math.sqrt(2)
    z = -(rows[:, None] + cols[None, :]) * step                 # tan = 1, aspect SE
    out = ls.ls_factor_desmet_govers(_pad(z), upa, np.zeros(5), RES, lambda_max=122)
    np.testing.assert_allclose(out, _reference_ls(sin_cap, 122 * D, D, x=math.sqrt(2)), rtol=1e-9)


def test_area_term_sca_is_glosem_compatible():
    dy = ls.EARTH_RADIUS_M * math.radians(RES)
    D = dy                                                      # equator: dx == dy
    tan_t = 0.2
    rows = np.arange(5)
    z = np.repeat((-rows * dy * tan_t)[:, None], 4, axis=1)    # south-facing ramp, x = 1
    upa = np.full((5, 4), 1.0)                                  # km2: above the 122 m cap
    sin_t = tan_t / math.sqrt(1 + tan_t ** 2)

    sca = ls.ls_factor_desmet_govers(_pad(z), upa, np.zeros(5), RES, lambda_max=122, area_term="sca")
    np.testing.assert_allclose(sca, _reference_ls(sin_t, 122.0, D), rtol=1e-9)   # A/(D x) = 122 "m2"
    area = ls.ls_factor_desmet_govers(_pad(z), upa, np.zeros(5), RES, lambda_max=122)
    np.testing.assert_allclose(area, _reference_ls(sin_t, 122 * D, D), rtol=1e-9)
    assert np.all(area / sca > 1.5)                             # the SAGA-style L is far lower

    # a hilltop cell (nothing drains into it) is the same in both modes
    top = np.full((5, 4), (D * D) / 1e6)
    np.testing.assert_allclose(
        ls.ls_factor_desmet_govers(_pad(z), top, np.zeros(5), RES, area_term="sca"),
        ls.ls_factor_desmet_govers(_pad(z), top, np.zeros(5), RES), rtol=1e-9)

    with pytest.raises(ValueError, match="area_term"):
        ls.ls_factor_desmet_govers(_pad(z), upa, np.zeros(5), RES, area_term="m2")


@pytest.mark.parametrize("tan_t, m, s_coef", [
    (0.005, 0.2, (10.8, 0.03)), (0.02, 0.3, (10.8, 0.03)), (0.05, 0.4, (10.8, 0.03)),
    (0.12, 0.5, (16.8, -0.50)), (0.3, 0.5, (21.9, -0.96)), (1.0, 0.5, (21.9, -0.96)),
])
def test_ls2_slope_classes_and_liu_s(tan_t, m, s_coef):
    dy = ls.EARTH_RADIUS_M * math.radians(RES)
    D = dy                                                      # equator: dx == dy
    rows = np.arange(5)
    z = np.repeat((-rows * dy * tan_t)[:, None], 4, axis=1)
    upa = np.full((5, 4), 1.0)                                  # km2: lambda capped at 122 m
    out = ls.ls_factor_desmet_govers(_pad(z), upa, np.zeros(5), RES, lambda_max=122, area_term="ls2")
    t = min(tan_t, math.tan(math.radians(26.6)))                # 45 deg is clamped like the other variants
    sin_t = t / math.sqrt(1 + t * t)
    np.testing.assert_allclose(out, (122 / 22.13) ** m * (s_coef[0] * sin_t + s_coef[1]), rtol=1e-9)

    # a hilltop cell: lambda = D (the cell itself); no cap: lambda = upa / D
    top = ls.ls_factor_desmet_govers(_pad(z), np.full((5, 4), D * D / 1e6), np.zeros(5), RES, area_term="ls2")
    np.testing.assert_allclose(top, (D / 22.13) ** m * (s_coef[0] * sin_t + s_coef[1]), rtol=1e-9)
    free = ls.ls_factor_desmet_govers(_pad(z), upa, np.zeros(5), RES, lambda_max=None, area_term="ls2")
    np.testing.assert_allclose(free, (1e6 / D / 22.13) ** m * (s_coef[0] * sin_t + s_coef[1]), rtol=1e-9)


def test_missing_neighbours_fall_back_to_one_sided():
    z = np.array([[0.0, 1.0, 4.0, 9.0]])                        # quadratic along x
    p = _pad(z)
    dzdx, _ = ls.slope_components(p, 1.0, np.ones(1))
    np.testing.assert_allclose(dzdx[0], [1.0, 2.0, 4.0, 5.0])  # edges one-sided, interior central


def test_aggregate_block_sums_and_weights():
    a = np.arange(16, dtype=float).reshape(4, 4)
    a[0, 0] = np.nan
    w = np.full((4, 4), 0.5)
    w[1, 1] = np.nan                                            # NaN weight where LS is valid: excluded from w
    s, n, sw, ws = ls.aggregate_block(a, w, factor=2)
    assert n[0, 0] == 3 and s[0, 0] == 1 + 4 + 5
    np.testing.assert_allclose(sw[0, 0], 0.5 * (1 + 4))
    np.testing.assert_allclose(ws[0, 0], 1.0)
    np.testing.assert_allclose(sw[1, 1], 0.5 * (10 + 11 + 14 + 15))

    s1, n1, sw1, ws1 = ls.aggregate_block(a, None, factor=1)    # factor 1 keeps the pixels
    np.testing.assert_array_equal(s1, np.nan_to_num(a))
    assert not sw1.any() and not ws1.any()
    with pytest.raises(ValueError):
        ls.aggregate_block(a[:3], factor=2)


def test_grid_helpers():
    west, south, east, north = ls._snap_bbox((20.01, 9.99, 20.02, 10.01), ls.FIVE_MIN_DEG)
    assert (west, east) == pytest.approx((20.0, 20 + 1 / 12))
    assert (south, north) == pytest.approx((10 - 1 / 12, 10 + 1 / 12))
    assert ls._snap_bbox((20.0, 10.0, 21.0, 11.0), ls.FIVE_MIN_DEG) == pytest.approx((20.0, 10.0, 21.0, 11.0))

    assert ls.merit_boxes_for_bbox((25, 40, 35, 45)) == ["n30e000", "n30e030"]
    assert ls.merit_boxes_for_bbox((-5, -35, 5, -25)) == ["s30w030", "s30e000", "s60w030", "s60e000"]
    assert ls.merit_boxes_for_bbox((0, 0, 30, 30)) == ["n00e000"]           # edges do not pull in neighbours
    assert ls.merit_boxes_for_bbox((0, -80, 1, -70)) == []                  # south of MERIT


def test_merit_box_names():
    names = ls.merit_box_names(lat_bands=(30, -30), lon_start=-30, lon_end=30)
    assert names == ["n30w030", "n30e000", "s30w030", "s30e000"]


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


@pytest.mark.parametrize("factor", [100, 10, 1])
def test_process_archive_stitches_tiles_exactly(tmp_path, factor):
    tars, size = _make_archives(tmp_path)
    out_fp = tmp_path / "parts" / "ls_sums_test.tif"
    sums, transform = ls.process_archive(tars["dem"], tars["upa"], out_fp, out_res_deg=factor * RES,
                                         block_rows=100, show_progress=False)

    ref = ls.aggregate_block(_full_mosaic_ls(size), None, factor)
    assert sums.shape == (4, 2 * TILE // factor, 2 * TILE // factor)
    np.testing.assert_allclose(sums[0], ref[0], rtol=1e-10)
    np.testing.assert_array_equal(sums[1], ref[1])
    assert not sums[2].any() and not sums[3].any()              # no crop file -> crop bands 0
    assert transform.c == pytest.approx(20.0) and transform.f == pytest.approx(10 + 2 * size)
    assert transform.a == pytest.approx(factor * RES)
    with rasterio.open(out_fp) as src:
        assert src.count == 4 and src.descriptions == ls.LS_BANDS


def test_one_sided_edges_only_change_cells_on_tile_lines(tmp_path):
    tars, _ = _make_archives(tmp_path)                          # 200 px tiles = 2 x 2 cells of 5' (100 px)
    sums, _ = ls.process_archive(tars["dem"], tars["upa"], None, block_rows=100, show_progress=False)
    one_sided, _ = ls.process_archive(tars["dem"], tars["upa"], None, block_rows=100,
                                      use_neighbours=False, show_progress=False)
    changed = ~np.isclose(one_sided[0], sums[0], rtol=1e-12)
    touches_inner_edge = np.zeros_like(changed)
    touches_inner_edge[[1, 2], :] = True                        # cells either side of the inner tile lines
    touches_inner_edge[:, [1, 2]] = True
    assert changed[touches_inner_edge].any()
    assert not changed[~touches_inner_edge].any()


def test_bbox_gives_the_matching_slice(tmp_path):
    tars, size = _make_archives(tmp_path)
    res = 10 * RES                                              # 30"
    full, tr_full = ls.process_archive(tars["dem"], tars["upa"], None, out_res_deg=res,
                                       block_rows=100, show_progress=False)
    bbox = (20 + size - 0.03, 10 + size - 0.04, 20 + size + 0.05, 10 + size + 0.02)   # crosses both tile lines
    part, tr = ls.process_archive(tars["dem"], tars["upa"], None, out_res_deg=res, bbox=bbox,
                                  block_rows=100, show_progress=False)

    for v in ((tr.c + 180) / res, (90 - tr.f) / res):           # origin on the global grid
        assert v == pytest.approx(round(v), abs=1e-6)
    assert tr.c <= bbox[0] and tr.f >= bbox[3]
    assert tr.c + part.shape[2] * res >= bbox[2] and tr.f - part.shape[1] * res <= bbox[1]
    c, r = round((tr.c - tr_full.c) / res), round((tr_full.f - tr.f) / res)
    np.testing.assert_allclose(part, full[:, r: r + part.shape[1], c: c + part.shape[2]], rtol=1e-12)

    empty, _ = ls.process_archive(tars["dem"], tars["upa"], None, bbox=(0, 0, 1, 1), show_progress=False)
    assert empty.size == 0


def test_crop_weights(tmp_path):
    tars, size = _make_archives(tmp_path)
    crop_fp = tmp_path / "crop.tif"
    n = 2 * TILE // 2                                           # crop raster at 6", covering the archive
    crop = np.zeros((n, n), dtype=np.float32)
    crop[:, : n // 2] = 100.0                                   # percent: west half all cropland
    with rasterio.open(crop_fp, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(20.0, 10 + 2 * size, 2 * RES, 2 * RES)) as dst:
        dst.write(crop, 1)

    sums, _ = ls.process_archive(tars["dem"], tars["upa"], None, crop_fp=crop_fp, crop_scale=0.01,
                                 block_rows=100, show_progress=False)
    np.testing.assert_allclose(sums[3][:, :2], sums[1][:, :2])  # west cells: w_sum == n
    np.testing.assert_allclose(sums[2][:, :2], sums[0][:, :2])
    assert not sums[3][:, 2:].any() and not sums[2][:, 2:].any()


def test_bad_resolution_and_memory_guard(tmp_path, monkeypatch):
    tars, _ = _make_archives(tmp_path)
    with pytest.raises(ValueError, match="multiple"):
        ls.process_archive(tars["dem"], tars["upa"], None, out_res_deg=1.5 * RES, show_progress=False)
    monkeypatch.setattr(ls, "MAX_SUM_BYTES", 1000)
    with pytest.raises(ValueError, match="bbox"):
        ls.process_archive(tars["dem"], tars["upa"], None, out_res_deg=RES, block_rows=100, show_progress=False)


# --- batch run and mosaic -----------------------------------------------------------------------------
def test_run_local_processes_present_boxes_and_keeps_tars(tmp_path, monkeypatch):
    tars, _ = _make_archives(tmp_path)
    archive_dir, parts = tars["dem"].parent, tmp_path / "parts"
    kw = dict(dem_name="dem_{box}.tar", upa_name="upa_{box}.tar", block_rows=100, show_progress=False)

    manifest, not_found = ls.run_local(archive_dir, parts, boxes=["test", "n00e000"], **kw)
    assert manifest["test"] == "done" and not_found == ["n00e000"] and "n00e000" not in manifest
    assert (parts / "ls_sums_test.tif").exists()
    assert tars["dem"].exists() and tars["upa"].exists()        # nothing deleted
    params = json.loads((parts / "manifest.json").read_text())["_params"]
    assert params["crop"] is False
    assert params["lambda_max"] == 122.0 and params["max_slope_deg"] == 26.6   # GloSEM v1.1 defaults
    assert "area_term" not in params                                          # default "area" is not recorded

    def _fail(*a, **k):
        raise AssertionError("done boxes must be skipped")
    monkeypatch.setattr(ls, "process_archive", _fail)
    ls.run_local(archive_dir, parts, boxes=["test"], **kw)
    with pytest.raises(ValueError, match="parts_dir"):
        ls.run_local(archive_dir, parts, boxes=["test"], lambda_max=100, **kw)
    with pytest.raises(ValueError, match="parts_dir"):                         # sca parts need their own folder
        ls.run_local(archive_dir, parts, boxes=["test"], area_term="sca", **kw)


def test_mosaic_places_parts_and_divides(tmp_path):
    parts = tmp_path / "parts"
    sums = np.zeros((4, 2, 3))
    sums[0], sums[1], sums[2], sums[3] = 6.0, 2.0, 3.0, 1.0
    ls.write_sums(parts / "ls_sums_a.tif", sums, from_origin(0.0, 0.0, ls.FIVE_MIN_DEG, ls.FIVE_MIN_DEG))
    ls_all, ls_crop = ls.mosaic_ls(parts, tmp_path / "all.tif", tmp_path / "crop.tif")
    assert ls_all.shape == (2160, 4320)
    r0, c0 = round(90 / ls.FIVE_MIN_DEG), round(180 / ls.FIVE_MIN_DEG)
    np.testing.assert_allclose(ls_all[r0:r0 + 2, c0:c0 + 3], 3.0)
    np.testing.assert_allclose(ls_crop[r0:r0 + 2, c0:c0 + 3], 3.0)
    assert np.isnan(ls_all[0, 0]) and np.isfinite(ls_all).sum() == 6
    _, none = ls.mosaic_ls(parts, tmp_path / "all.tif")
    assert none is None


def test_mosaic_on_a_bbox_grid(tmp_path):
    res = 10 * RES
    parts = tmp_path / "parts"
    sums = np.zeros((4, 2, 3))
    sums[0], sums[1] = 6.0, 2.0
    ls.write_sums(parts / "ls_sums_a.tif", sums, from_origin(10.0, 5.0, res, res))
    out, _ = ls.mosaic_ls(parts, tmp_path / "all.tif", res_deg=res, bbox=(10.0, 4.9, 10.1, 5.0))
    assert out.shape == (12, 12)
    np.testing.assert_allclose(out[:2, :3], 3.0)
    assert np.isfinite(out).sum() == 6
    with rasterio.open(tmp_path / "all.tif") as src:
        assert src.transform.c == pytest.approx(10.0) and src.transform.f == pytest.approx(5.0)


def test_mosaic_rejects_bad_parts(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    old = parts / "ls_sums_old.tif"
    with rasterio.open(old, "w", driver="GTiff", height=1, width=1, count=2, dtype="float64",
                       crs="EPSG:4326", transform=from_origin(10, 10, ls.FIVE_MIN_DEG, ls.FIVE_MIN_DEG)) as dst:
        dst.write(np.ones((2, 1, 1)))
    with pytest.raises(ValueError, match="bands"):
        ls.mosaic_ls(parts, tmp_path / "ls.tif")
    old.unlink()

    ls.write_sums(parts / "ls_sums_b.tif", np.ones((4, 1, 1)), from_origin(10, 10, RES * 10, RES * 10))
    with pytest.raises(ValueError, match="resolution"):
        ls.mosaic_ls(parts, tmp_path / "ls.tif")
    (parts / "ls_sums_b.tif").unlink()

    no_crop = np.ones((4, 1, 1))
    no_crop[2:] = 0.0
    ls.write_sums(parts / "ls_sums_c.tif", no_crop, from_origin(10, 10, ls.FIVE_MIN_DEG, ls.FIVE_MIN_DEG))
    with pytest.raises(ValueError, match="crop"):
        ls.mosaic_ls(parts, tmp_path / "ls.tif", tmp_path / "crop.tif")


# --- several area terms in one pass ---------------------------------------------------------------------------
def test_tuple_area_term_matches_single_terms(tmp_path):
    lon, lat = np.meshgrid(20 + (np.arange(30) + 0.5) * RES, 10 - (np.arange(20) + 0.5) * RES)
    z = _surface(lon, lat) * 20                                  # steep enough for both S branches and the clamp
    upa = np.random.default_rng(0).uniform(0.005, 2.0, z.shape)
    lat_rows = lat[:, 0]
    both = ls.ls_factor_desmet_govers(_pad(z), upa, lat_rows, RES, area_term=("area", "sca", "ls2"))
    for term in ("area", "sca", "ls2"):
        np.testing.assert_array_equal(both[term], ls.ls_factor_desmet_govers(_pad(z), upa, lat_rows, RES,
                                                                             area_term=term))
    with pytest.raises(ValueError, match="area_term"):
        ls.ls_factor_desmet_govers(_pad(z), upa, lat_rows, RES, area_term=("area", "area"))

    tars, _ = _make_archives(tmp_path)
    kw = dict(block_rows=100, show_progress=False)
    out = {"area": tmp_path / "a" / "p.tif", "sca": tmp_path / "s" / "p.tif"}
    sums, tr = ls.process_archive(tars["dem"], tars["upa"], out, area_term=("area", "sca"), **kw)
    for term in ("area", "sca"):
        single, tr1 = ls.process_archive(tars["dem"], tars["upa"], None, area_term=term, **kw)
        np.testing.assert_array_equal(sums[term], single)
        with rasterio.open(out[term]) as src:
            np.testing.assert_array_equal(src.read(), single)
    assert tr == tr1
    with pytest.raises(ValueError, match="out_fp"):
        ls.process_archive(tars["dem"], tars["upa"], tmp_path / "x.tif", area_term=("area", "sca"), **kw)


def test_run_local_tuple_area_term_writes_both_folders(tmp_path):
    tars, _ = _make_archives(tmp_path)
    archive_dir = tars["dem"].parent
    kw = dict(dem_name="dem_{box}.tar", upa_name="upa_{box}.tar", block_rows=100, show_progress=False)
    dirs = {"area": tmp_path / "parts_area", "sca": tmp_path / "parts_sca", "ls2": tmp_path / "parts_ls2"}
    manifests, not_found = ls.run_local(archive_dir, dirs, boxes=["test", "n00e000"], area_term=tuple(dirs), **kw)
    assert not_found == ["n00e000"] and all(m["test"] == "done" for m in manifests.values())

    for term, d in dirs.items():                                  # same parts and manifest as a single-term run
        single_dir = tmp_path / f"single_{term}"
        single, _ = ls.run_local(archive_dir, single_dir, boxes=["test"], area_term=term, **kw)
        assert json.loads((d / "manifest.json").read_text()) == json.loads((single_dir / "manifest.json").read_text())
        with rasterio.open(d / "ls_sums_test.tif") as a, rasterio.open(single_dir / "ls_sums_test.tif") as b:
            np.testing.assert_array_equal(a.read(), b.read())
        ls.run_local(archive_dir, d, boxes=["test"], area_term=term, **kw)   # folders stay usable one term at a time

    with pytest.raises(ValueError, match="parts_dir"):
        ls.run_local(archive_dir, tmp_path / "one", boxes=["test"], area_term=("area", "sca"), **kw)
