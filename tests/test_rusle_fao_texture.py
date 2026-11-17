"""Tests for the FAO texture class (Jones, Spoor & Thomasson 2003, Table 1) and the structure class s in sbtn_leaf.rusle."""

import numpy as np

from sbtn_leaf import rusle


def test_fao_texture_class_boundaries():
    #               coarse  coarse-edge medium medium(>65 sand, 18 clay) medium fine fine  fine-edge very fine
    sand = np.array([80,    66,         40,    70,                       10,         30,   30,       10], dtype=np.float32)
    clay = np.array([10,    17,         20,    18,                       30,         40,   60,       65], dtype=np.float32)
    assert rusle._determine_fao_texture_class(sand, clay).tolist() == [1, 1, 2, 2, 3, 4, 4, 5]


def test_fao_texture_class_nodata():
    sand = np.array([np.nan, 50], dtype=np.float32)
    clay = np.array([20, 20], dtype=np.float32)
    assert rusle._determine_fao_texture_class(sand, clay).tolist() == [0, 2]
    assert rusle._determine_fao_texture_class(sand, clay, valid=np.array([True, False])).tolist() == [0, 0]


def test_structure_class_from_usda_function():
    # sand, silt, clay -> FAO coarse, medium, medium fine, fine, very fine, nodata
    sand = np.array([85, 40, 10, 30, 10, 0], dtype=np.float32)
    silt = np.array([10, 40, 70, 30, 25, 0], dtype=np.float32)
    clay = np.array([5, 20, 20, 40, 65, 0], dtype=np.float32)
    texture, s, p = rusle._determine_usda_class_number_permeability(sand, silt, clay)
    assert s.tolist() == [1, 2, 2, 4, 4, 0]
    assert s.dtype == np.uint8 and p.dtype == np.uint8
    assert texture[-1] == 0 and p[-1] == 0


def _write(fp, data, dtype, nodata, crs="ESRI:54052"):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(fp, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=dtype,
                       nodata=nodata, crs=crs, transform=from_origin(0, 3000, 1000, 1000)) as dst:
        dst.write(data.astype(dtype), 1)


def test_calculate_K_RUSLE_2_structure_switch(tmp_path):
    import pytest
    import rasterio

    # 1 x 3 pixels, SoilGrids units (g/kg, dg/kg). Pixel 1: fine (clay 40%) -> texture s = 4
    lists = {}
    for name, vals in (("clay", [400, 200, 100]), ("silt", [300, 400, 100]), ("sand", [300, 400, 800]), ("soc", [100, 100, 100])):
        lists[name] = []
        for d in range(3):
            fp = tmp_path / f"{name}{d}.tif"
            _write(fp, np.array([vals]), "int16", -32768)
            lists[name].append(fp)
    s_hwsd_fp = tmp_path / "s_hwsd.tif"
    _write(s_hwsd_fp, np.array([[2, 2, 0]]), "uint8", 0)   # HWSD: Normal, Normal, no soil

    kw = dict(clay_fp_list=lists["clay"], silt_fp_list=lists["silt"], sand_fp_list=lists["sand"], soc_fp_list=lists["soc"])
    rusle.calculate_K_RUSLE_2(tmp_path / "k_tex.tif", **kw)
    rusle.calculate_K_RUSLE_2(tmp_path / "k_hwsd.tif", structure_method="hwsd", s_hwsd_fp=s_hwsd_fp, **kw)
    k_tex = rasterio.open(tmp_path / "k_tex.tif").read(1)[0]
    k_hwsd = rasterio.open(tmp_path / "k_hwsd.tif").read(1)[0]

    # Pixel 1: s 4 -> 2 changes Ks by 3.25 * 2 -> K by 0.1317 * 6.5 / 100
    assert k_hwsd[0] - k_tex[0] == pytest.approx(0.1317 * 6.5 / 100, rel=1e-4)
    # Pixel 2: both s = 2. Pixel 3: no HWSD soil -> falls back to texture s
    assert k_hwsd[1] == pytest.approx(k_tex[1]) and k_hwsd[2] == pytest.approx(k_tex[2])

    with pytest.raises(ValueError):
        rusle.calculate_K_RUSLE_2(tmp_path / "x.tif", structure_method="other", **kw)
