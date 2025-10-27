import numpy as np
import numpy.testing as npt
import xarray as xr
import geopandas as gpd
import polars as pl


_ORIGINAL_READ_FILE = gpd.read_file


def _safe_read_file(path, *args, **kwargs):
    if isinstance(path, str) and path.startswith("../data/"):
        return gpd.GeoDataFrame()
    return _ORIGINAL_READ_FILE(path, *args, **kwargs)


gpd.read_file = _safe_read_file

_ORIGINAL_READ_EXCEL = pl.read_excel


def _safe_read_excel(source, *args, **kwargs):
    if isinstance(source, str) and source.endswith("forest_residues_IPCC.xlsx"):
        return pl.DataFrame(
            {
                "IPCC Climate": ["Temperate"],
                "BD_mean": [1.0],
                "NE_mean": [1.0],
                "BD_TP": [20],
                "NE_TP": [20],
            }
        )
    return _ORIGINAL_READ_EXCEL(source, *args, **kwargs)


pl.read_excel = _safe_read_excel

from sbtn_leaf.RothC_Raster import (
    raster_rothc_annual_results_1yrloop,
    raster_rothc_ReducedTillage_annual_results_1yrloop,
    run_RothC_forest,
)


def test_reduced_tillage_trm_uses_full_soc(monkeypatch):
    n_years = 1
    y = x = 2
    months = 12

    clay = np.full((y, x), 30.0, dtype=float)
    soc0 = np.full((y, x), 50.0, dtype=float)
    tmp = np.full((months, y, x), 15.0, dtype=float)
    rain = np.full((months, y, x), 80.0, dtype=float)
    evap = np.full((months, y, x), 20.0, dtype=float)
    pc = np.ones((months, y, x), dtype=int)
    sand = np.full((y, x), 40.0, dtype=float)

    call_counter = {"count": 0}

    def fake_trm(sand_arr, soc_arr):
        call_counter["count"] += 1
        assert sand_arr.shape == (y, x)
        assert soc_arr.shape == (y, x)
        # return neutral modifiers so dynamics remain stable
        ones = np.ones_like(sand_arr, dtype=float)
        return ones, ones, ones, ones

    monkeypatch.setattr(
        "sbtn_leaf.RothC_Raster.RMF_TRM",
        fake_trm,
    )

    soc_annual, co2_annual = raster_rothc_ReducedTillage_annual_results_1yrloop(
        n_years=n_years,
        clay=clay,
        soc0=soc0,
        tmp=tmp,
        rain=rain,
        evap=evap,
        pc=pc,
        sand=sand,
        commodity_type="annual_crop",
    )

    # TRM should be applied once per monthly timestep (n_years * 12)
    assert call_counter["count"] == n_years * months

    # Output shapes should remain consistent with expectations
    assert soc_annual.shape == (n_years + 1, y, x)
    assert co2_annual.shape == (n_years + 1, y, x)


def test_reduced_tillage_trm_accepts_time_varying_sand(monkeypatch):
    n_years = 1
    y = x = 2
    months = 12

    clay = np.full((y, x), 30.0, dtype=float)
    soc0 = np.full((y, x), 50.0, dtype=float)
    tmp = np.full((months, y, x), 15.0, dtype=float)
    rain = np.full((months, y, x), 80.0, dtype=float)
    evap = np.full((months, y, x), 20.0, dtype=float)
    pc = np.ones((months, y, x), dtype=int)
    sand = np.full((months, y, x), 40.0, dtype=float)

    captured_shapes = []

    def fake_trm(sand_arr, soc_arr):
        captured_shapes.append((sand_arr.shape, soc_arr.shape))
        ones = np.ones_like(sand_arr, dtype=float)
        return ones, ones, ones, ones

    monkeypatch.setattr(
        "sbtn_leaf.RothC_Raster.RMF_TRM",
        fake_trm,
    )

    raster_rothc_ReducedTillage_annual_results_1yrloop(
        n_years=n_years,
        clay=clay,
        soc0=soc0,
        tmp=tmp,
        rain=rain,
        evap=evap,
        pc=pc,
        sand=sand,
        commodity_type="annual_crop",
    )

    # Every call should receive matching 2-D slices for sand and the full SOC state
    assert len(captured_shapes) == n_years * months
    assert all(s == ((y, x), (y, x)) for s in captured_shapes)


def test_raster_rothc_baseline_regression(monkeypatch):
    monkeypatch.setattr("sbtn_leaf.RothC_Raster.trange", lambda n, **_: range(n))

    n_years = 2
    y = x = 2
    months = 12

    clay = np.full((y, x), 25.0, dtype=float)
    soc0 = np.full((y, x), 40.0, dtype=float)
    tmp = np.full((months, y, x), 15.0, dtype=float)
    rain = np.full((months, y, x), 80.0, dtype=float)
    evap = np.full((months, y, x), 20.0, dtype=float)
    pc = np.ones((months, y, x), dtype=float)
    irr = np.zeros_like(tmp)

    expected_soc = np.array(
        [
            [[40.0, 40.0], [40.0, 40.0]],
            [[37.534622, 37.534622], [37.534622, 37.534622]],
            [[35.909515, 35.909515], [35.909515, 35.909515]],
        ],
        dtype=np.float32,
    )
    expected_co2 = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[2.465377, 2.465377], [2.465377, 2.465377]],
            [[1.6251065, 1.6251065], [1.6251065, 1.6251065]],
        ],
        dtype=np.float32,
    )

    soc, co2 = raster_rothc_annual_results_1yrloop(
        n_years=n_years,
        clay=clay,
        soc0=soc0,
        tmp=tmp,
        rain=rain,
        evap=evap,
        pc=pc,
        irr=irr,
        commodity_type="annual_crop",
    )

    npt.assert_allclose(soc, expected_soc)
    npt.assert_allclose(co2, expected_co2)


def test_raster_rothc_reduced_tillage_regression(monkeypatch):
    monkeypatch.setattr("sbtn_leaf.RothC_Raster.trange", lambda n, **_: range(n))

    n_years = 2
    y = x = 2
    months = 12

    clay = np.full((y, x), 25.0, dtype=float)
    soc0 = np.full((y, x), 40.0, dtype=float)
    tmp = np.full((months, y, x), 15.0, dtype=float)
    rain = np.full((months, y, x), 80.0, dtype=float)
    evap = np.full((months, y, x), 20.0, dtype=float)
    pc = np.ones((months, y, x), dtype=float)
    irr = np.zeros_like(tmp)
    sand = np.full((y, x), 30.0, dtype=float)

    expected_soc = np.array(
        [
            [[40.0, 40.0], [40.0, 40.0]],
            [[37.60099, 37.60099], [37.60099, 37.60099]],
            [[36.023407, 36.023407], [36.023407, 36.023407]],
        ],
        dtype=np.float32,
    )
    expected_co2 = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[2.3990104, 2.3990104], [2.3990104, 2.3990104]],
            [[1.5775822, 1.5775822], [1.5775822, 1.5775822]],
        ],
        dtype=np.float32,
    )

    soc, co2 = raster_rothc_ReducedTillage_annual_results_1yrloop(
        n_years=n_years,
        clay=clay,
        soc0=soc0,
        tmp=tmp,
        rain=rain,
        evap=evap,
        pc=pc,
        sand=sand,
        irr=irr,
        commodity_type="annual_crop",
    )

    npt.assert_allclose(soc, expected_soc)
    npt.assert_allclose(co2, expected_co2)


def test_run_rothc_forest_handles_single_band_age(monkeypatch, tmp_path):
    monkeypatch.setattr("sbtn_leaf.RothC_Raster.trange", lambda n, **_: range(n))

    n_years = 2
    y = x = 2
    months = 12

    coords = {
        "time": np.arange(months),
        "y": np.arange(y),
        "x": np.arange(x),
    }

    tmp = xr.DataArray(np.full((months, y, x), 12.0, dtype=float), dims=("time", "y", "x"), coords=coords)
    rain = xr.full_like(tmp, 80.0)
    evap = xr.full_like(tmp, 20.0)
    pc = xr.full_like(tmp, 1.0)

    soc0 = xr.DataArray(np.full((y, x), 50.0, dtype=float), dims=("y", "x"))
    clay = xr.full_like(soc0, 30.0)
    iom = xr.full_like(soc0, 5.0)
    sand = xr.full_like(soc0, 40.0)
    lu = xr.full_like(soc0, 1.0)
    evap_da = xr.DataArray(np.full((months, y, x), 20.0, dtype=float), dims=("time", "y", "x"), coords=coords)
    pc_da = xr.full_like(tmp, 1.0)
    age = xr.DataArray(np.full((1, y, x), 10.0, dtype=float), dims=("band", "y", "x"))

    monkeypatch.setattr(
        "sbtn_leaf.RothC_Raster._load_environmental_data",
        lambda *_: (tmp, rain, soc0, iom, clay, sand),
    )
    monkeypatch.setattr(
        "sbtn_leaf.RothC_Raster._load_forest_data",
        lambda *_: (lu, evap_da, pc_da, age),
    )
    monkeypatch.setattr("sbtn_leaf.RothC_Raster.save_annual_results", lambda *_, **__: None)

    def fake_forest_litter(age_arr, *_, **__):
        y_dim, x_dim = age_arr.shape
        return np.full((1, y_dim, x_dim), 0.5, dtype=float)

    monkeypatch.setattr(
        "sbtn_leaf.RothC_Raster.cropcalcs.get_forest_litter_monthlyrate_fromda",
        fake_forest_litter,
    )

    soc = run_RothC_forest(
        forest_type="BRDC",
        weather_type="Temperate",
        n_years=n_years,
        save_folder=str(tmp_path),
        data_description="test",
        lu_fp="lu.tif",
        evap_fp="evap.tif",
        age_fp="age.tif",
    )

    assert soc.shape == (n_years + 1, y, x)
    assert np.isfinite(soc).all()
