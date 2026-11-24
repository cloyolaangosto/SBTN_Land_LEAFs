# RUSLE LS factor from MERIT DEM + MERIT Hydro, processed one 30-degree archive at a time
'''
LS (slope length and steepness) for the new RUSLE layers, method LS1 of RUSLE/README.md:

- L: Desmet & Govers (1996) two-dimensional unit contributing area, with the McCool et al. (1989)
  rill/interrill exponent (as GloSEM, Borrelli et al. 2022).
- S: McCool et al. (1987).
- Inputs: MERIT DEM (elevation) and MERIT Hydro `upa` (upstream drainage area), both 3 arc-second.

MERIT is distributed as 30x30 degree tar archives of 36 tiles of 5x5 degrees (6000 x 6000 px). The work unit
here is one archive pair (DEM tar + upa tar of the same box), downloaded and deleted by hand. The tiles are read
in place through GDAL /vsitar/ (nothing is extracted), and a small result of sums is written per box. This module
never downloads or deletes anything.

Output grid:
- Lat/lon (EPSG:4326), any resolution that is a whole multiple of 3": 5' (default, factor 100), 30" ~ 1 km
  (factor 10), 3" ~ 90 m (factor 1, no aggregation). Metric grids are a later step on the finished rasters.
- The grid is anchored at (-180, 90), so parts from different boxes, or different bboxes, always mosaic exactly.
- An optional bbox (west, south, east, north) limits the work and the output to a region; it is snapped outward to
  the grid. Fine resolutions need one: a full 30-deg box at 3" would need ~41 GB of sums.

Edges:
- upa is MERIT Hydro's global flow accumulation, so there is no flow-routing edge problem.
- Slope uses central differences (Zevenbergen & Thorne 1987, the SAGA default used by GloSEM). The stencil only
  needs the 4 direct neighbours, so tile borders are stitched from one row or column of the N/S/W/E tiles.
- Where a neighbour is missing (outer edge of the archive, coast, nodata) the difference falls back to one-sided.
- 5 deg = 6000 x 3" and every allowed factor divides 6000, so no output cell straddles two tiles.

Each archive writes sums and counts (LS_sum, n, LSw_sum, w_sum) so the mosaic is exact:
- LS_all = LS_sum / n: mean over all valid land pixels.
- LS_crop = LSw_sum / w_sum: weighted by cropland fraction (GloSEM 1.3 ESA100_Cropland_fraction.tif).
LS per pixel is terrain only; the weighting acts only in the aggregation. Crops tend to sit on the flatter part of
a cell, and GloSEM computes erosion on cropland pixels only, so both layers are kept for testing.
'''
# SETUP
## Module imports
import json
import math
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from tqdm.auto import tqdm

PathLike = Union[str, Path]
BBox = Tuple[float, float, float, float]  # west, south, east, north (degrees)

## Constants
EARTH_RADIUS_M = 6_371_007.2            # authalic sphere radius, m
UNIT_PLOT_LENGTH_M = 22.13              # RUSLE unit plot length, m
FIVE_MIN_DEG = 1.0 / 12.0               # 5 arc-minutes
LS_BANDS = ("LS_sum", "n", "LSw_sum", "w_sum")
MAX_SUM_BYTES = 4 * 2**30               # memory guard for the in-memory sums array

DEFAULT_LAMBDA_MAX_M = 300.0            # slope-length cap (README: sensitivity 100-500 m)
MERIT_LAT_BANDS = (60, 30, 0, -30, -60) # south edge of each 30-deg archive row (MERIT covers 60S-90N)
_EPS = 1e-6                             # tolerance, in grid cells, when snapping degrees to the grid


# LS FORMULAS
def _central_or_one_sided(prev: np.ndarray, centre: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Elevation difference per pixel step along one axis.

    Central difference (nxt - prev) / 2 where both neighbours exist, one-sided where only one exists, 0 where none.
    """
    has_prev, has_next = np.isfinite(prev), np.isfinite(nxt)
    with np.errstate(invalid="ignore"):
        return np.where(
            has_prev & has_next, (nxt - prev) * 0.5,
            np.where(has_next, nxt - centre, np.where(has_prev, centre - prev, 0.0)),
        )


def slope_components(z_pad: np.ndarray, dy_m: float, dx_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """dz/dx (east) and dz/dy (north) for the inner cells of a 1-pixel padded elevation array.

    Parameters
    ----------
    z_pad: (n+2, m+2) elevations in m, NaN where missing. Corners are never used.
    dy_m: pixel height in m.
    dx_m: (n,) pixel width in m for each inner row.
    """
    centre = z_pad[1:-1, 1:-1]
    dzdx = _central_or_one_sided(z_pad[1:-1, :-2], centre, z_pad[1:-1, 2:]) / dx_m[:, None]
    # rows run north -> south, so "previous" (south) is the row below
    dzdy = _central_or_one_sided(z_pad[2:, 1:-1], centre, z_pad[:-2, 1:-1]) / dy_m
    return dzdx, dzdy


def ls_factor_desmet_govers(
    z_pad: np.ndarray,
    upa_km2: np.ndarray,
    lat_deg: np.ndarray,
    res_deg: float,
    lambda_max: Optional[float] = DEFAULT_LAMBDA_MAX_M,
    channel_upa_km2: Optional[float] = None,
) -> np.ndarray:
    """RUSLE LS for a block of a geographic (lat/lon) grid.

    L (Desmet & Govers 1996):
        L = ((A + D^2)^(m+1) - A^(m+1)) / (D^(m+2) * x^m * 22.13^m)
        A = upslope contributing area at the cell inlet (m^2) = upa * 1e6 - D^2, capped at lambda_max * D
        m = beta / (1 + beta), beta = (sin(theta) / 0.0896) / (3 sin(theta)^0.8 + 0.56)   (McCool et al. 1989)
        x = |sin(aspect)| + |cos(aspect)|
    S (McCool et al. 1987): 10.8 sin(theta) + 0.03 if tan(theta) < 0.09, else 16.8 sin(theta) - 0.50

    Parameters
    ----------
    z_pad: (n+2, m+2) elevation (m) with a 1-pixel border from the neighbouring tiles; NaN where missing.
    upa_km2: (n, m) MERIT Hydro upstream drainage area, km^2 (includes the cell itself); NaN where missing.
    lat_deg: (n,) latitude of each row centre.
    res_deg: pixel size in degrees (1/1200 for MERIT 3").
    lambda_max: cap on the slope length A/D, m. None disables the cap.
    channel_upa_km2: cells with upa above this are treated as channels and set to NaN. None keeps all cells.

    Returns
    -------
    (n, m) LS, NaN where elevation or upa is missing (or channel cells, if masked).
    """
    res_rad = math.radians(res_deg)
    dy = EARTH_RADIUS_M * res_rad
    dx = EARTH_RADIUS_M * res_rad * np.cos(np.radians(lat_deg))
    D = np.sqrt(dx * dy)[:, None]                                  # cell size, m (rows differ with latitude)

    dzdx, dzdy = slope_components(z_pad, dy, dx)
    grad = np.hypot(dzdx, dzdy)                                    # tan(theta)
    sin_t = grad / np.sqrt(1.0 + grad * grad)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(grad > 0, (np.abs(dzdx) + np.abs(dzdy)) / grad, 1.0)

    A = np.maximum(upa_km2 * 1e6 - D * D, 0.0)
    if lambda_max is not None:
        A = np.minimum(A, lambda_max * D)

    beta = (sin_t / 0.0896) / (3.0 * sin_t ** 0.8 + 0.56)
    m = beta / (1.0 + beta)
    L = ((A + D * D) ** (m + 1) - A ** (m + 1)) / (D ** (m + 2) * x ** m * UNIT_PLOT_LENGTH_M ** m)
    S = np.where(grad < 0.09, 10.8 * sin_t + 0.03, 16.8 * sin_t - 0.50)

    ls = L * S
    invalid = ~np.isfinite(z_pad[1:-1, 1:-1]) | ~np.isfinite(upa_km2)
    if channel_upa_km2 is not None:
        invalid |= upa_km2 > channel_upa_km2
    ls[invalid] = np.nan
    return ls


def aggregate_block(ls: np.ndarray, weight: Optional[np.ndarray] = None, factor: int = 100):
    """Block sums of a 3" block to output cells of factor x factor pixels (factor 1 keeps the pixels).

    Returns LS_sum, n (valid pixel count), LSw_sum (sum of weight*LS) and w_sum (sum of weight where LS valid).
    A weight counts only where LS is valid and the weight is finite. Both dimensions of `ls` must be multiples
    of `factor`.
    """
    nr, nc = ls.shape
    if nr % factor or nc % factor:
        raise ValueError(f"block shape {ls.shape} is not a multiple of {factor}")
    valid = np.isfinite(ls)
    ls0 = np.where(valid, ls, 0.0)
    w0 = np.zeros_like(ls0) if weight is None else np.where(valid & np.isfinite(weight), weight, 0.0)

    def _bsum(a):
        return a.reshape(nr // factor, factor, nc // factor, factor).sum(axis=(1, 3))

    return _bsum(ls0), _bsum(valid.astype(np.float64)), _bsum(ls0 * w0), _bsum(w0)


# OUTPUT GRID
def _grid_index(bbox: BBox, res: float) -> Tuple[int, int, int, int]:
    """Column and row bounds (c0, c1, r0, r1) of a bbox on the global grid anchored at (-180, 90), snapped outward."""
    west, south, east, north = bbox
    if not (west < east and south < north):
        raise ValueError(f"bbox must be (west, south, east, north), got {bbox}")
    return (math.floor((west + 180.0) / res + _EPS), math.ceil((east + 180.0) / res - _EPS),
            math.floor((90.0 - north) / res + _EPS), math.ceil((90.0 - south) / res - _EPS))


def _snap_bbox(bbox: BBox, res: float) -> BBox:
    """bbox grown outward to the global `res` grid anchored at (-180, 90)."""
    c0, c1, r0, r1 = _grid_index(bbox, res)
    return (-180.0 + c0 * res, 90.0 - r1 * res, -180.0 + c1 * res, 90.0 - r0 * res)


def _box_label(lat: int, lon: int) -> str:
    return f"{'n' if lat >= 0 else 's'}{abs(lat):02d}{'e' if lon >= 0 else 'w'}{abs(lon):03d}"


def merit_box_names(lat_bands: Iterable[int] = MERIT_LAT_BANDS, lon_start: int = -180, lon_end: int = 180) -> List[str]:
    """Archive box labels such as 'n30e000' (south-west corner), for every 30-deg box in the given bands."""
    return [_box_label(lat, lon) for lat in lat_bands for lon in range(lon_start, lon_end, 30)]


def merit_boxes_for_bbox(bbox: BBox) -> List[str]:
    """Labels of the 30-deg MERIT boxes that overlap a bbox (north to south, west to east)."""
    west, south, east, north = bbox
    if not (west < east and south < north):
        raise ValueError(f"bbox must be (west, south, east, north), got {bbox}")
    lats = [lat for lat in MERIT_LAT_BANDS if lat < north and lat + 30 > south]
    lons = [lon for lon in range(-180, 180, 30) if lon < east and lon + 30 > west]
    return [_box_label(lat, lon) for lat in lats for lon in lons]


# MERIT ARCHIVES
def list_tar_tifs(tar_fp: PathLike) -> List[str]:
    """GeoTIFF members of a tar archive (reads only the headers)."""
    with tarfile.open(tar_fp) as tf:
        return [m.name for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".tif")]


def _vsitar(tar_fp: PathLike, member: str) -> str:
    return f"/vsitar/{Path(tar_fp).resolve().as_posix()}/{member}"


class _TileSet:
    """Open tiles of one archive, keyed by their north-west corner in pixel units."""

    def __init__(self, tar_fp: PathLike):
        self.tar_fp = Path(tar_fp)
        self.tiles: Dict[Tuple[int, int], rasterio.io.DatasetReader] = {}
        self.names: Dict[Tuple[int, int], str] = {}
        res = None
        for member in list_tar_tifs(tar_fp):
            ds = rasterio.open(_vsitar(tar_fp, member))
            r = ds.res[0]
            if res is None:
                res = r
            elif not math.isclose(r, res, rel_tol=1e-9):
                raise ValueError(f"{member}: resolution {r} differs from {res}")
            key = (round(ds.bounds.left / r), round(ds.bounds.top / r))
            self.tiles[key] = ds
            self.names[key] = Path(member).stem
        if not self.tiles:
            raise ValueError(f"no GeoTIFF tiles in {tar_fp}")
        self.res = res
        first = next(iter(self.tiles.values()))
        self.height, self.width = first.height, first.width

    def by_name(self) -> Dict[str, Tuple[int, int]]:
        return {name: key for key, name in self.names.items()}

    def neighbour(self, key, d_col: int, d_row: int):
        """Tile d_col tiles east and d_row tiles south of `key`, or None."""
        return self.tiles.get((key[0] + d_col * self.width, key[1] - d_row * self.height))

    def origin_px(self, key) -> Tuple[int, int]:
        """(column, row) of the tile's north-west pixel on the global 3" grid anchored at (-180, 90)."""
        return key[0] + round(180.0 / self.res), round(90.0 / self.res) - key[1]

    def close(self):
        for ds in self.tiles.values():
            ds.close()


def _read_float(ds, window: Window) -> np.ndarray:
    a = ds.read(1, window=window).astype(np.float64)
    if ds.nodata is not None:
        a[a == ds.nodata] = np.nan
    return a


def _read_padded(ts: _TileSet, key, r0: int, r1: int, c0: int, c1: int, use_neighbours: bool = True) -> np.ndarray:
    """Rows r0:r1, columns c0:c1 of tile `key` with a 1-pixel border.

    Border pixels come from the same tile where they exist, otherwise from the N/S/W/E tile. Corners are not filled.
    """
    ds, H, W = ts.tiles[key], ts.height, ts.width
    pad = np.full((r1 - r0 + 2, c1 - c0 + 2), np.nan)
    lo, hi = max(r0 - 1, 0), min(r1 + 1, H)
    clo, chi = max(c0 - 1, 0), min(c1 + 1, W)
    pad[lo - (r0 - 1): hi - (r0 - 1), clo - (c0 - 1): chi - (c0 - 1)] = _read_float(ds, Window(clo, lo, chi - clo, hi - lo))
    if not use_neighbours:
        return pad
    if r0 == 0 and (north := ts.neighbour(key, 0, -1)) is not None:
        pad[0, 1:-1] = _read_float(north, Window(c0, H - 1, c1 - c0, 1))[0]
    if r1 == H and (south := ts.neighbour(key, 0, 1)) is not None:
        pad[-1, 1:-1] = _read_float(south, Window(c0, 0, c1 - c0, 1))[0]
    if c0 == 0 and (west := ts.neighbour(key, -1, 0)) is not None:
        pad[1:-1, 0] = _read_float(west, Window(W - 1, r0, 1, r1 - r0))[:, 0]
    if c1 == W and (east := ts.neighbour(key, 1, 0)) is not None:
        pad[1:-1, -1] = _read_float(east, Window(0, r0, 1, r1 - r0))[:, 0]
    return pad


def _crop_weight(crop_src, tile_ds, scale: float, window: Window) -> np.ndarray:
    """Cropland fraction (0-1) resampled onto a window of the 3" tile grid; NaN where missing."""
    dst = np.full((int(window.height), int(window.width)), np.nan, dtype=np.float32)
    reproject(
        source=rasterio.band(crop_src, 1), destination=dst,
        dst_transform=window_transform(window, tile_ds.transform), dst_crs=tile_ds.crs, dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return dst.astype(np.float64) * scale


def process_archive(
    dem_tar: PathLike,
    upa_tar: PathLike,
    out_fp: Optional[PathLike] = None,
    crop_fp: Optional[PathLike] = None,
    crop_scale: float = 1.0,
    out_res_deg: float = FIVE_MIN_DEG,
    bbox: Optional[BBox] = None,
    lambda_max: Optional[float] = DEFAULT_LAMBDA_MAX_M,
    channel_upa_km2: Optional[float] = None,
    block_rows: int = 1200,
    use_neighbours: bool = True,
    only_tiles: Optional[Sequence[str]] = None,
    show_progress: bool = True,
):
    """LS sums and counts on the output grid for one MERIT archive pair.

    Parameters
    ----------
    dem_tar, upa_tar: MERIT DEM and MERIT Hydro upa tar archives of the same box (not extracted, never deleted).
    out_fp: 4-band GeoTIFF to write (bands LS_sum, n, LSw_sum, w_sum). None returns the arrays only. Nothing is
        written when the bbox misses the archive.
    crop_fp: cropland-fraction raster (e.g. GloSEM 1.3 ESA100_Cropland_fraction.tif). None leaves the crop bands 0.
    crop_scale: multiplier that turns crop_fp values into a 0-1 fraction (0.01 if the file is in percent).
    out_res_deg: output cell size in degrees; a whole multiple of the 3" pixel (5' default, 1/120 for 30", 1/1200
        for 3").
    bbox: (west, south, east, north) in degrees; snapped outward to the output grid. None covers the whole archive.
    lambda_max, channel_upa_km2: see ls_factor_desmet_govers.
    block_rows: rows read at a time; must be a multiple of the aggregation factor.
    use_neighbours: stitch tile borders from neighbouring tiles. False gives one-sided differences at every
        tile edge (used to measure the archive-edge effect).
    only_tiles: tile names (file stems in the DEM tar, e.g. "n45e005_dem") to process; None processes all.

    Returns
    -------
    (sums, transform): sums is a (4, rows, cols) float64 array on the output grid (the archive's extent, or its
    intersection with the bbox). With no overlap it has 0 rows and columns.
    """
    dem, upa = _TileSet(dem_tar), _TileSet(upa_tar)
    crop_src = rasterio.open(crop_fp) if crop_fp is not None else None
    try:
        res, H, W = dem.res, dem.height, dem.width
        factor = round(out_res_deg / res)
        if factor < 1 or not math.isclose(factor * res, out_res_deg, rel_tol=1e-9):
            raise ValueError(f"out_res_deg {out_res_deg} is not a whole multiple of the pixel size {res}")
        if H % factor or W % factor or block_rows % factor:
            raise ValueError(f"tile size {H}x{W} and block_rows {block_rows} must be multiples of factor {factor}")

        # archive extent and output grid, in global 3" pixel indices
        origins = {key: dem.origin_px(key) for key in dem.tiles}
        gx0, gy0 = min(o[0] for o in origins.values()), min(o[1] for o in origins.values())
        gx1, gy1 = max(o[0] for o in origins.values()) + W, max(o[1] for o in origins.values()) + H
        if any(v % factor for v in (gx0, gy0, gx1, gy1)):
            raise ValueError(f"archive edges are not on the {out_res_deg}-degree output grid")
        if bbox is not None:
            c0, c1, r0, r1 = _grid_index(bbox, out_res_deg)
            gx0, gx1 = max(gx0, c0 * factor), min(gx1, c1 * factor)
            gy0, gy1 = max(gy0, r0 * factor), min(gy1, r1 * factor)
        ncols, nrows = max(gx1 - gx0, 0) // factor, max(gy1 - gy0, 0) // factor
        transform = from_origin(-180.0 + gx0 * res, 90.0 - gy0 * res, out_res_deg, out_res_deg)
        nbytes = len(LS_BANDS) * nrows * ncols * 8
        if nbytes > MAX_SUM_BYTES:
            raise ValueError(f"output grid {nrows}x{ncols} needs {nbytes / 2**30:.1f} GB; pass a smaller bbox")
        sums = np.zeros((len(LS_BANDS), nrows, ncols))
        if nrows == 0 or ncols == 0:
            return sums, transform

        keys = list(dem.tiles)
        if only_tiles is not None:
            names = dem.by_name()
            missing = [t for t in only_tiles if t not in names]
            if missing:
                raise KeyError(f"tiles not in {dem_tar}: {missing}")
            keys = [names[t] for t in only_tiles]

        for key in tqdm(keys, desc=dem.tar_fp.name, disable=not show_progress):
            upa_ds = upa.tiles.get(key)
            if upa_ds is None:
                continue                                           # no upa for this tile: LS undefined
            tx0, ty0 = origins[key]
            wc0, wc1 = max(gx0, tx0) - tx0, min(gx1, tx0 + W) - tx0
            wr0, wr1 = max(gy0, ty0) - ty0, min(gy1, ty0 + H) - ty0
            if wc1 <= wc0 or wr1 <= wr0:
                continue                                           # tile outside the bbox
            tile_ds = dem.tiles[key]
            weight = (_crop_weight(crop_src, tile_ds, crop_scale, Window(wc0, wr0, wc1 - wc0, wr1 - wr0))
                      if crop_src is not None else None)
            tile_north = key[1] * res
            oc = (tx0 + wc0 - gx0) // factor
            for r0 in range(wr0, wr1, block_rows):
                r1 = min(r0 + block_rows, wr1)
                z_pad = _read_padded(dem, key, r0, r1, wc0, wc1, use_neighbours)
                upa_blk = _read_float(upa_ds, Window(wc0, r0, wc1 - wc0, r1 - r0))
                lat = tile_north - (np.arange(r0, r1) + 0.5) * res
                ls = ls_factor_desmet_govers(z_pad, upa_blk, lat, res, lambda_max, channel_upa_km2)
                w = weight[r0 - wr0: r1 - wr0] if weight is not None else None
                blk = aggregate_block(ls, w, factor)
                orow = (ty0 + r0 - gy0) // factor
                for b in range(len(LS_BANDS)):
                    sums[b, orow: orow + blk[b].shape[0], oc: oc + blk[b].shape[1]] += blk[b]
    finally:
        dem.close()
        upa.close()
        if crop_src is not None:
            crop_src.close()

    if out_fp is not None:
        write_sums(out_fp, sums, transform)
    return sums, transform


def write_sums(out_fp: PathLike, sums: np.ndarray, transform) -> None:
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=sums.shape[1], width=sums.shape[2], count=len(LS_BANDS), dtype="float64",
                   crs="EPSG:4326", transform=transform, compress="deflate")
    with rasterio.open(out_fp, "w", **profile) as dst:
        dst.write(sums)
        for b, name in enumerate(LS_BANDS, start=1):
            dst.set_band_description(b, name)


# BATCH RUN OVER LOCAL ARCHIVES
def _load_manifest(fp: Path) -> Dict:
    return json.loads(fp.read_text()) if fp.exists() else {}


def _run_params(bbox: Optional[BBox], process_kwargs: Dict) -> Dict:
    """Settings that must stay the same for every part in one parts_dir (JSON round-tripped for comparison)."""
    params = dict(
        out_res_deg=process_kwargs.get("out_res_deg", FIVE_MIN_DEG),
        bbox=list(bbox) if bbox is not None else None,
        lambda_max=process_kwargs.get("lambda_max", DEFAULT_LAMBDA_MAX_M),
        channel_upa_km2=process_kwargs.get("channel_upa_km2"),
        crop=process_kwargs.get("crop_fp") is not None,
    )
    return json.loads(json.dumps(params))


def run_local(
    archive_dir: PathLike,
    parts_dir: PathLike,
    boxes: Optional[Sequence[str]] = None,
    bbox: Optional[BBox] = None,
    dem_name: str = "dem_tif_{box}.tar",
    upa_name: str = "upa_{box}.tar",
    **process_kwargs,
) -> Tuple[Dict, List[str]]:
    """Process every box whose DEM and upa tars are in archive_dir; resumable through parts_dir/manifest.json.

    The tars are downloaded and deleted by hand; this function never downloads or deletes anything.

    Parameters
    ----------
    archive_dir: folder holding the downloaded tar pairs.
    parts_dir: where ls_sums_{box}.tif and manifest.json go. One parts_dir holds one set of run settings
        (resolution, bbox, lambda_max, channel threshold, crop or not); a different set raises.
    boxes: box labels to look for (e.g. "n30e000"). None uses the boxes overlapping `bbox`, or all MERIT boxes.
    bbox: (west, south, east, north) passed to process_archive.
    dem_name, upa_name: tar file names, with {box} for the label.
    process_kwargs: passed to process_archive (crop_fp, crop_scale, out_res_deg, lambda_max, ...).

    Returns
    -------
    (manifest, not_found): manifest maps box -> "done" or "error: ..." (errors are retried on the next call);
    not_found lists boxes whose tars are not in archive_dir yet (not recorded, so they are picked up later).
    """
    archive_dir, parts_dir = Path(archive_dir), Path(parts_dir)
    if boxes is None:
        boxes = merit_boxes_for_bbox(bbox) if bbox is not None else merit_box_names()
    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest_fp = parts_dir / "manifest.json"
    manifest = _load_manifest(manifest_fp)

    params = _run_params(bbox, process_kwargs)
    if "_params" in manifest and manifest["_params"] != params:
        raise ValueError(f"{manifest_fp} was made with {manifest['_params']}, not {params}; use a new parts_dir")
    manifest["_params"] = params

    not_found = []
    for box in boxes:
        if manifest.get(box) == "done":
            continue
        dem_tar, upa_tar = archive_dir / dem_name.format(box=box), archive_dir / upa_name.format(box=box)
        if not (dem_tar.exists() and upa_tar.exists()):
            not_found.append(box)
            continue
        try:
            sums, transform = process_archive(dem_tar, upa_tar, None, bbox=bbox, **process_kwargs)
            if sums.size:
                write_sums(parts_dir / f"ls_sums_{box}.tif", sums, transform)
            manifest[box] = "done"
        except Exception as exc:                                   # keep going; the box is retried next call
            manifest[box] = f"error: {exc}"
        finally:
            manifest_fp.write_text(json.dumps(manifest, indent=1))
    return manifest, not_found


# MOSAIC
def mosaic_ls(
    parts_dir: PathLike,
    out_all_fp: PathLike,
    out_crop_fp: Optional[PathLike] = None,
    res_deg: float = FIVE_MIN_DEG,
    bbox: Optional[BBox] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Add every ls_sums_*.tif into one grid and write LS_all = LS_sum/n and LS_crop = LSw_sum/w_sum.

    The grid has cell size res_deg and covers the bbox (snapped outward), or the globe when bbox is None
    (4320 x 2160 at 5'). out_crop_fp None skips LS_crop.
    """
    c0, c1, r0, r1 = _grid_index(bbox if bbox is not None else (-180.0, -90.0, 180.0, 90.0), res_deg)
    width, height = c1 - c0, r1 - r0
    if len(LS_BANDS) * width * height * 8 > MAX_SUM_BYTES:
        raise ValueError(f"mosaic grid {height}x{width} is too large; pass a bbox")
    sums = np.zeros((len(LS_BANDS), height, width))
    for fp in sorted(Path(parts_dir).glob("ls_sums_*.tif")):
        with rasterio.open(fp) as src:
            if src.count != len(LS_BANDS):
                raise ValueError(f"{fp}: {src.count} bands, expected {LS_BANDS} (stale part from an older run?)")
            if not all(math.isclose(r, res_deg, rel_tol=1e-9) for r in src.res):
                raise ValueError(f"{fp}: resolution {src.res} is not {res_deg}")
            pc, pr = (src.transform.c + 180.0) / res_deg, (90.0 - src.transform.f) / res_deg
            if abs(pc - round(pc)) > _EPS or abs(pr - round(pr)) > _EPS:
                raise ValueError(f"{fp}: origin is not on the {res_deg}-degree grid")
            part = src.read()
        pc, pr = round(pc) - c0, round(pr) - r0
        rows = slice(max(pr, 0), min(pr + part.shape[1], height))
        cols = slice(max(pc, 0), min(pc + part.shape[2], width))
        if rows.stop <= rows.start or cols.stop <= cols.start:
            continue                                               # part outside the mosaic bbox
        sums[:, rows, cols] += part[:, rows.start - pr: rows.stop - pr, cols.start - pc: cols.stop - pc]

    if out_crop_fp is not None and not sums[3].any():
        raise ValueError("no crop weights in the parts (were they made without crop_fp?)")
    with np.errstate(invalid="ignore", divide="ignore"):
        ls_all = np.where(sums[1] > 0, sums[0] / sums[1], np.nan).astype(np.float32)
        ls_crop = np.where(sums[3] > 0, sums[2] / sums[3], np.nan).astype(np.float32)

    profile = dict(driver="GTiff", height=height, width=width, count=1, dtype="float32", nodata=np.nan,
                   crs="EPSG:4326", transform=from_origin(-180.0 + c0 * res_deg, 90.0 - r0 * res_deg, res_deg, res_deg),
                   compress="deflate")
    targets = [(out_all_fp, ls_all)] + ([(out_crop_fp, ls_crop)] if out_crop_fp is not None else [])
    for fp, arr in targets:
        Path(fp).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(fp, "w", **profile) as dst:
            dst.write(arr, 1)
    return ls_all, (ls_crop if out_crop_fp is not None else None)
