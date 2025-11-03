import importlib


def test_import_package():
    """Basic smoke test: can import the package and check version."""
    pkg = importlib.import_module("sbtn_leaf")

    assert hasattr(pkg, "__version__")
    assert pkg.__version__.startswith("0.")


def test_import_functions():
    """Check that key functions are exposed at top-level."""
    import sbtn_leaf as sl

    assert hasattr(sl, "__version__")
    assert hasattr(sl, "__author__")

    expected_exports = {
        "calculate_PET_crop_based",
        "run_simulation",
        "raster_rothc_annual_results_1yrloop",
        "calculate_area_weighted_cfs_from_shp_with_std_and_median",
        "create_crop_yield_raster",
        "plot_raster_on_world_extremes_cutoff",
    }

    for name in expected_exports:
        assert hasattr(sl, name), f"Expected '{name}' to be re-exported"

    assert expected_exports.issubset(set(sl.__all__))
