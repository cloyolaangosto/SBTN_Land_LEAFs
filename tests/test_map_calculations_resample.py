"""Regression tests for the consolidated raster resampling helper."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling

from sbtn_leaf.map_calculations import resample_raster_to_match


def _write_raster(path: Path, data: np.ndarray, transform, crs: str = "EPSG:4326", nodata=None) -> None:
    data = np.asarray(data)
    if data.ndim == 2:
        bands = 1
        height, width = data.shape
    elif data.ndim == 3:
        bands, height, width = data.shape
    else:
        raise ValueError("Raster data must be 2D or 3D.")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": data.dtype,
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
    }

    with rasterio.open(path, "w", **profile) as dst:
        if bands == 1:
            dst.write(data, 1)
        else:
            dst.write(data)


def test_resample_raster_single_band(tmp_path):
    src_path = tmp_path / "src_single.tif"
    target_path = tmp_path / "target_single.tif"
    output_path = tmp_path / "resampled_single.tif"

    src_data = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    src_transform = from_origin(0, 3, 1, 1)
    _write_raster(src_path, src_data, src_transform, nodata=-9999.0)

    target_transform = from_origin(0, 3, 0.5, 0.5)
    _write_raster(target_path, np.zeros((6, 6), dtype=np.float32), target_transform, nodata=-9999.0)

    resample_raster_to_match(
        str(src_path),
        str(target_path),
        output_path=str(output_path),
        resampling_method=Resampling.nearest,
        src_nodata=-9999.0,
        dst_nodata=-9999.0,
    )

    with rasterio.open(output_path) as dst:
        assert dst.count == 1
        assert dst.height == 6 and dst.width == 6
        data = dst.read(1)

    expected = np.repeat(np.repeat(src_data, 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(data, expected)


def test_resample_raster_multiband(tmp_path):
    src_path = tmp_path / "src_multi.tif"
    target_path = tmp_path / "target_multi.tif"
    output_path = tmp_path / "resampled_multi.tif"

    band1 = np.array([[1, 2], [3, 4]], dtype=np.int16)
    band2 = np.array([[5, 6], [7, 8]], dtype=np.int16)
    src_data = np.stack([band1, band2])
    src_transform = from_origin(0, 2, 1, 1)
    _write_raster(src_path, src_data, src_transform)

    target_transform = from_origin(0, 2, 0.5, 0.5)
    _write_raster(target_path, np.zeros((4, 4), dtype=np.int16), target_transform)

    resample_raster_to_match(
        str(src_path),
        str(target_path),
        bands=[1, 2],
        output_path=str(output_path),
        resampling_method=Resampling.nearest,
    )

    with rasterio.open(output_path) as dst:
        assert dst.count == 2
        data = dst.read()

    expected_band1 = np.repeat(np.repeat(band1, 2, axis=0), 2, axis=1)
    expected_band2 = np.repeat(np.repeat(band2, 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(data[0], expected_band1)
    np.testing.assert_array_equal(data[1], expected_band2)


def test_resample_raster_returns_array(tmp_path):
    src_path = tmp_path / "src_array.tif"
    target_path = tmp_path / "target_array.tif"

    src_data = np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    src_transform = from_origin(0, 2, 1, 1)
    _write_raster(src_path, src_data, src_transform, nodata=0.0)

    target_transform = from_origin(0, 2, 1, 1)
    _write_raster(target_path, np.zeros((2, 2), dtype=np.float32), target_transform, nodata=-1.0)

    result = resample_raster_to_match(
        str(src_path),
        str(target_path),
        dst_nodata=-1.0,
        src_nodata=0.0,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    expected = np.array([[1.0, -1.0], [3.0, 4.0]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
