"""Tests for the HWSD v2 soil structure (s) and profile permeability (p) classes in sbtn_leaf.rusle_hwsd2."""

import sqlite3

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform

from sbtn_leaf import rusle_hwsd2 as rusle


def _soils(**columns):
    """Soil table with the columns of rusle._load_HWSD2_soils, filled with neutral defaults."""
    n = len(next(iter(columns.values())))
    defaults = {
        "smu": np.arange(1, n + 1),
        "seq": np.ones(n, dtype=int),
        "share": np.full(n, 100),
        "fao90": ["CMe"] * n,
        "phase1": [np.nan] * n,
        "phase2": [np.nan] * n,
        "il": [np.nan] * n,
        "drainage": ["MW"] * n,
        "p_texture": [3.0] * n,
    }
    defaults.update(columns)
    return pd.DataFrame(defaults)


def test_structure_class_follows_esdb_rule():
    soils = _soils(fao90=["PHh", "CMe", "HSf", "WR", "LPk", "RK", None])

    cultivated = rusle._HWSD2_structure_class(soils, "cultivated")
    grass = rusle._HWSD2_structure_class(soils, "grass_seminatural")

    # Good soil names -> 1, other soils -> 2, Histosols -> 4, water / rock / missing -> 0
    assert cultivated.tolist() == [1, 2, 4, 0, 1, 0, 0]
    # Under grass or semi-natural land every soil is Good, except the peaty ones
    assert grass.tolist() == [1, 1, 4, 0, 1, 0, 0]


def test_structure_class_rejects_unknown_land_use():
    with pytest.raises(ValueError):
        rusle._HWSD2_structure_class(_soils(fao90=["CMe"]), "forest")


def test_permeability_class_restricting_layers_and_drainage():
    soils = _soils(
        fao90=["CMe", "LVh", "LVh", "GLe", "WR", "PLe"],
        p_texture=[2.0, 3.0, 4.0, 3.0, 0.0, 3.0],
        il=[np.nan, 4, 1, np.nan, np.nan, np.nan],          # 4 = impermeable layer within 40 cm, 1 = none
        phase1=[np.nan, np.nan, 8, np.nan, np.nan, np.nan],  # 8 = fragipan
        phase2=[np.nan, np.nan, np.nan, np.nan, np.nan, 6],  # 6 = petroferric
        drainage=["MW", "MW", "MW", "P", "MW", "MW"],
    )

    texture_only = rusle._HWSD2_permeability_class(soils, use_restricting_layers=False)
    default = rusle._HWSD2_permeability_class(soils)
    with_drainage = rusle._HWSD2_permeability_class(soils, use_drainage=True)

    assert texture_only.tolist() == [2, 3, 4, 3, 0, 3]
    assert default.tolist() == [2, 6, 6, 3, 0, 6]
    assert with_drainage.tolist() == [2, 6, 6, 6, 0, 6]
    # the inputs are not modified
    assert soils["p_texture"].tolist() == [2.0, 3.0, 4.0, 3.0, 0.0, 3.0]


def test_dominant_class_uses_summed_share_and_ignores_non_soils():
    soils = _soils(
        smu=[1, 1, 1, 2, 2, 3, 4, 4],
        seq=[1, 2, 3, 1, 2, 1, 1, 2],
        share=[40, 30, 30, 50, 50, 100, 70, 30],
    )
    soil_class = np.array([2, 1, 1, 4, 2, 0, 0, 2], dtype=np.uint8)

    lookup = rusle._HWSD2_dominant_class(soils, soil_class)

    assert lookup[1] == 1   # class 1 covers 60% in two soils, against 40% for class 2
    assert lookup[2] == 4   # tie: the class of the soil listed first
    assert lookup[3] == 0   # no soil in the unit
    assert lookup[4] == 2   # the non-soil part (70%) is left out
    assert lookup[rusle.HWSD2_NODATA] == 0


def _write_test_inputs(tmp_path):
    """A 40 x 4 km reference grid in Homolosine around 8E 50N, an HWSD-like raster and its database."""
    crs = "ESRI:54052"
    xs, ys = transform("EPSG:4326", crs, [8.0], [50.0])
    ref_fp = tmp_path / "ref.tif"
    ref_transform = from_origin(xs[0] - 20000, ys[0] + 2000, 1000, 1000)
    with rasterio.open(ref_fp, "w", driver="GTiff", height=4, width=40, count=1, dtype="int16", nodata=-32768, crs=crs, transform=ref_transform) as dst:
        dst.write(np.ones((4, 40), dtype=np.int16), 1)

    # 30 arc-second grid over 7.5-8.5E, 49.5-50.5N: mapping unit 10 west of 8E, 20 east of it
    res = 1 / 120
    smu = np.full((120, 120), 10, dtype=np.uint16)
    smu[:, 60:] = 20
    hwsd_fp = tmp_path / "hwsd.tif"
    with rasterio.open(hwsd_fp, "w", driver="GTiff", height=120, width=120, count=1, dtype="uint16", nodata=65535, crs="EPSG:4326", transform=from_origin(7.5, 50.5, res, res)) as dst:
        dst.write(smu, 1)

    db_fp = tmp_path / "hwsd.sqlite"
    con = sqlite3.connect(db_fp)
    con.execute("CREATE TABLE HWSD2_LAYERS (HWSD2_SMU_ID INTEGER, SEQUENCE INTEGER, SHARE INTEGER, FAO90 TEXT, PHASE1 INTEGER, PHASE2 INTEGER, IL INTEGER, DRAINAGE TEXT, TOPDEP INTEGER, TEXTURE_USDA INTEGER)")
    con.executemany(
        "INSERT INTO HWSD2_LAYERS VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # unit 10: a Cambisol (60%) with a sandy loam topsoil over clay, and water (40%)
            (10, 1, 60, "CMe", None, None, None, "MW", 0, 11),
            (10, 1, 60, "CMe", None, None, None, "MW", 40, 3),
            (10, 2, 40, "WR", None, None, None, None, 0, None),
            # unit 20: a Histosol with a loam texture
            (20, 1, 100, "HSf", None, None, None, "VP", 0, 9),
        ],
    )
    con.commit()
    con.close()
    return ref_fp, hwsd_fp, db_fp


def test_s_and_p_rasters_end_to_end(tmp_path):
    ref_fp, hwsd_fp, db_fp = _write_test_inputs(tmp_path)
    smu_grid_fp = tmp_path / "smu_1km.tif"
    s_fp, p_fp, p_top_fp = tmp_path / "s.tif", tmp_path / "p.tif", tmp_path / "p_top.tif"

    rusle.create_HWSD2_SMU_grid(smu_grid_fp, ref_fp=ref_fp, hwsd_raster_fp=hwsd_fp)
    rusle.calculate_s_HWSD2(s_fp, smu_grid_fp=smu_grid_fp, db_fp=db_fp)
    rusle.calculate_p_HWSD2(p_fp, smu_grid_fp=smu_grid_fp, db_fp=db_fp)
    rusle.calculate_p_HWSD2(p_top_fp, profile_depth_cm=20, smu_grid_fp=smu_grid_fp, db_fp=db_fp)

    with rasterio.open(ref_fp) as ref, rasterio.open(smu_grid_fp) as smu, rasterio.open(s_fp) as s, rasterio.open(p_fp) as p, rasterio.open(p_top_fp) as p_top:
        for src in (smu, s, p, p_top):
            assert (src.crs, src.transform, src.shape) == (ref.crs, ref.transform, ref.shape)
        smu_data, s_data, p_data, p_top_data = smu.read(1), s.read(1), p.read(1), p_top.read(1)
        assert s.nodata == 0 and p.nodata == 0 and s.dtypes[0] == "uint8"

    # West end of the grid is unit 10, east end is unit 20
    assert set(np.unique(smu_data)) == {10, 20}
    assert (smu_data[:, 0] == 10).all() and (smu_data[:, -1] == 20).all()

    west, east = smu_data == 10, smu_data == 20
    assert (s_data[west] == 2).all() and (s_data[east] == 4).all()         # Normal, peaty
    assert (p_data[west] == 6).all() and (p_data[east] == 3).all()         # clay below 40 cm, loam
    assert (p_top_data[west] == 2).all() and (p_top_data[east] == 3).all()  # topsoil only: sandy loam, loam


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
