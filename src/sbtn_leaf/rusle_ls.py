# RUSLE LS factor from MERIT DEM + MERIT Hydro, processed one 30-degree archive at a time
'''
LS (slope length and steepness) for the new RUSLE layers, method LS1 of RUSLE/README.md:

- L: Desmet & Govers (1996) two-dimensional unit contributing area, with the McCool et al. (1989)
  rill/interrill exponent (as GloSEM, Borrelli et al. 2022).
- S: McCool et al. (1987).
- Inputs: MERIT DEM (elevation) and MERIT Hydro `upa` (upstream drainage area), both 3 arc-second.

MERIT is distributed as 30x30 degree tar archives of 36 tiles of 5x5 degrees (6000 x 6000 px). The work unit
here is one archive pair (DEM tar + upa tar of the same box): download, process in place through GDAL /vsitar/
(nothing is extracted), write a small 5-arc-minute result, delete the tars.

Edges:
- upa is MERIT Hydro's global flow accumulation, so there is no flow-routing edge problem.
- Slope uses central differences (Zevenbergen & Thorne 1987, the SAGA default used by GloSEM). The stencil only
  needs the 4 direct neighbours, so tile borders are stitched from one row or column of the N/S/W/E tiles.
- Where a neighbour is missing (outer edge of the archive, coast, nodata) the difference falls back to one-sided.
- 5 deg = 60 x 5' and 5' = 100 x 3", so every 5' cell lies inside one tile.

Each archive writes sums and counts (LS_sum, n) so the global mosaic is exact.

LS is a property of the terrain only, so the layer is land-use independent: one 5' mean over all valid land
pixels, shared by every LEAF flow (crops, grassland, forest, urban). Land-use differences enter through C.
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
from rasterio.windows import Window
from tqdm.auto import tqdm

PathLike = Union[str, Path]

## Constants
EARTH_RADIUS_M = 6_371_007.2            # authalic sphere radius, m
UNIT_PLOT_LENGTH_M = 22.13              # RUSLE unit plot length, m
FIVE_MIN_DEG = 1.0 / 12.0               # 5 arc-minutes
LS_BANDS = ("LS_sum", "n")

DEFAULT_LAMBDA_MAX_M = 300.0            # slope-length cap (README: sensitivity 100-500 m)
MERIT_LAT_BANDS = (60, 30, 0, -30, -60) # south edge of each 30-deg archive row (MERIT covers 60S-90N)


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


def aggregate_to_5min(ls: np.ndarray, factor: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Block sums of a 3" block to 5' cells (factor x factor pixels).

    Returns LS_sum and n (valid pixel count). Both dimensions of `ls` must be multiples of `factor`.
    """
    nr, nc = ls.shape
    if nr % factor or nc % factor:
        raise ValueError(f"block shape {ls.shape} is not a multiple of {factor}")
    valid = np.isfinite(ls)

    def _bsum(a):
        return a.reshape(nr // factor, factor, nc // factor, factor).sum(axis=(1, 3))

    return _bsum(np.where(valid, ls, 0.0)), _bsum(valid.astype(np.float64))


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

    def close(self):
        for ds in self.tiles.values():
            ds.close()


def _read_float(ds, window: Window) -> np.ndarray:
    a = ds.read(1, window=window).astype(np.float64)
    if ds.nodata is not None:
        a[a == ds.nodata] = np.nan
    return a


def _read_padded(ts: _TileSet, key, r0: int, r1: int, use_neighbours: bool = True) -> np.ndarray:
    """Rows r0:r1 of tile `key` with a 1-pixel border (rows from N/S tiles, columns from W/E tiles)."""
    ds, H, W = ts.tiles[key], ts.height, ts.width
    pad = np.full((r1 - r0 + 2, W + 2), np.nan)
    lo, hi = max(r0 - 1, 0), min(r1 + 1, H)
    pad[lo - (r0 - 1): hi - (r0 - 1), 1:W + 1] = _read_float(ds, Window(0, lo, W, hi - lo))
    if not use_neighbours:
        return pad
    north, south = ts.neighbour(key, 0, -1), ts.neighbour(key, 0, 1)
    west, east = ts.neighbour(key, -1, 0), ts.neighbour(key, 1, 0)
    if r0 == 0 and north is not None:
        pad[0, 1:W + 1] = _read_float(north, Window(0, H - 1, W, 1))[0]
    if r1 == H and south is not None:
        pad[-1, 1:W + 1] = _read_float(south, Window(0, 0, W, 1))[0]
    if west is not None:
        pad[1:-1, 0] = _read_float(west, Window(W - 1, r0, 1, r1 - r0))[:, 0]
    if east is not None:
        pad[1:-1, -1] = _read_float(east, Window(0, r0, 1, r1 - r0))[:, 0]
    return pad


def process_archive(
    dem_tar: PathLike,
    upa_tar: PathLike,
    out_fp: Optional[PathLike] = None,
    lambda_max: Optional[float] = DEFAULT_LAMBDA_MAX_M,
    channel_upa_km2: Optional[float] = None,
    block_rows: int = 1200,
    use_neighbours: bool = True,
    only_tiles: Optional[Sequence[str]] = None,
    show_progress: bool = True,
):
    """LS sums and counts on the 5' grid for one MERIT archive pair.

    Parameters
    ----------
    dem_tar, upa_tar: MERIT DEM and MERIT Hydro upa tar archives of the same box (not extracted).
    out_fp: 2-band GeoTIFF to write (bands LS_sum, n). None returns the arrays only.
    lambda_max, channel_upa_km2: see ls_factor_desmet_govers.
    block_rows: rows read at a time; must divide the tile height and be a multiple of 100.
    use_neighbours: stitch tile borders from neighbouring tiles. False gives one-sided differences at every
        tile edge (used to measure the archive-edge effect).
    only_tiles: tile names (file stems in the DEM tar, e.g. "n45e005_dem") to process; None processes all.

    Returns
    -------
    (sums, transform): sums is a (2, rows, cols) float64 array on the 5' grid of the archive's bounding box.
    """
    dem, upa = _TileSet(dem_tar), _TileSet(upa_tar)
    try:
        res, H, W = dem.res, dem.height, dem.width
        factor = round(FIVE_MIN_DEG / res)
        if not math.isclose(factor * res, FIVE_MIN_DEG, rel_tol=1e-9) or H % block_rows or block_rows % factor:
            raise ValueError("tile size, block_rows and the 5' grid do not line up")

        keys = list(dem.tiles)
        if only_tiles is not None:
            names = dem.by_name()
            missing = [t for t in only_tiles if t not in names]
            if missing:
                raise KeyError(f"tiles not in {dem_tar}: {missing}")
            keys = [names[t] for t in only_tiles]

        # 5' grid of the archive's bounding box
        west = min(k[0] for k in dem.tiles) * res
        north = max(k[1] for k in dem.tiles) * res
        east = (max(k[0] for k in dem.tiles) + W) * res
        south = (min(k[1] for k in dem.tiles) - H) * res
        ncols, nrows = round((east - west) / FIVE_MIN_DEG), round((north - south) / FIVE_MIN_DEG)
        transform = from_origin(west, north, FIVE_MIN_DEG, FIVE_MIN_DEG)
        sums = np.zeros((len(LS_BANDS), nrows, ncols))

        for key in tqdm(keys, desc=dem.tar_fp.name, disable=not show_progress):
            upa_ds = upa.tiles.get(key)
            if upa_ds is None:
                continue                                           # no upa for this tile: LS undefined
            tile_north = key[1] * res
            c0 = round((key[0] * res - west) / FIVE_MIN_DEG)
            for r0 in range(0, H, block_rows):
                r1 = r0 + block_rows
                z_pad = _read_padded(dem, key, r0, r1, use_neighbours)
                upa_blk = _read_float(upa_ds, Window(0, r0, W, block_rows))
                lat = tile_north - (np.arange(r0, r1) + 0.5) * res
                ls = ls_factor_desmet_govers(z_pad, upa_blk, lat, res, lambda_max, channel_upa_km2)
                blk = aggregate_to_5min(ls, factor)
                rr = round((north - tile_north) / FIVE_MIN_DEG) + r0 // factor
                for b in range(len(LS_BANDS)):
                    sums[b, rr: rr + block_rows // factor, c0: c0 + W // factor] += blk[b]
    finally:
        dem.close()
        upa.close()

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


# DOWNLOAD AND BATCH RUN
def merit_box_names(lat_bands: Iterable[int] = MERIT_LAT_BANDS, lon_start: int = -180, lon_end: int = 180) -> List[str]:
    """Archive box labels such as 'n30e000' (south-west corner), for every 30-deg box in the given bands."""
    names = []
    for lat in lat_bands:
        for lon in range(lon_start, lon_end, 30):
            names.append(f"{'n' if lat >= 0 else 's'}{abs(lat):02d}{'e' if lon >= 0 else 'w'}{abs(lon):03d}")
    return names


def download_file(url: str, dest: PathLike, auth: Optional[Tuple[str, str]] = None, chunk_mb: int = 8) -> bool:
    """Stream `url` to `dest`. Returns False on HTTP 404 (box has no data, e.g. ocean), raises on other errors."""
    import requests

    dest = Path(dest)
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with requests.get(url, auth=auth, stream=True, timeout=120) as r:
        if r.status_code == 404:
            return False
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with open(part, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as bar:
            for chunk in r.iter_content(chunk_size=chunk_mb * 2**20):
                f.write(chunk)
                bar.update(len(chunk))
    part.replace(dest)
    return True


def _load_manifest(fp: Path) -> Dict[str, str]:
    return json.loads(fp.read_text()) if fp.exists() else {}


def run_all(
    boxes: Sequence[str],
    dem_url_template: str,
    upa_url_template: str,
    work_dir: PathLike,
    parts_dir: PathLike,
    auth: Optional[Tuple[str, str]] = None,
    keep_archives: bool = False,
    **process_kwargs,
) -> Dict[str, str]:
    """Download, process and delete each archive pair; resumable through parts_dir/manifest.json.

    URL templates take the box label, e.g. ".../dem_tif_{box}.tar". Manifest status per box: "done",
    "missing" (no DEM or no upa archive on the server) or "error: ...".
    """
    work_dir, parts_dir = Path(work_dir), Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest_fp = parts_dir / "manifest.json"
    manifest = _load_manifest(manifest_fp)

    for box in boxes:
        if manifest.get(box) in ("done", "missing"):
            continue
        dem_tar = work_dir / Path(dem_url_template.format(box=box)).name
        upa_tar = work_dir / Path(upa_url_template.format(box=box)).name
        try:
            if not (download_file(dem_url_template.format(box=box), dem_tar, auth)
                    and download_file(upa_url_template.format(box=box), upa_tar, auth)):
                manifest[box] = "missing"
            else:
                process_archive(dem_tar, upa_tar, parts_dir / f"ls_sums_{box}.tif", **process_kwargs)
                manifest[box] = "done"
        except Exception as exc:                                   # keep going; the box can be retried later
            manifest[box] = f"error: {exc}"
        finally:
            if not keep_archives:
                for fp in (dem_tar, upa_tar):
                    fp.unlink(missing_ok=True)
            manifest_fp.write_text(json.dumps(manifest, indent=1))
    return manifest


# MOSAIC
def mosaic_ls(
    parts_dir: PathLike,
    out_fp: PathLike,
    west: float = -180.0,
    north: float = 90.0,
    width: int = 4320,
    height: int = 2160,
) -> np.ndarray:
    """Add every ls_sums_*.tif into one 5' grid and write LS = LS_sum / n (mean over all valid land pixels)."""
    sums = np.zeros((len(LS_BANDS), height, width))
    for fp in sorted(Path(parts_dir).glob("ls_sums_*.tif")):
        with rasterio.open(fp) as src:
            if src.count != len(LS_BANDS):
                raise ValueError(f"{fp}: {src.count} bands, expected {LS_BANDS} (stale part from an older run?)")
            part = src.read()
            c0 = round((src.transform.c - west) / FIVE_MIN_DEG)
            r0 = round((north - src.transform.f) / FIVE_MIN_DEG)
        rows = slice(max(r0, 0), min(r0 + part.shape[1], height))
        cols = slice(max(c0, 0), min(c0 + part.shape[2], width))
        sums[:, rows, cols] += part[:, rows.start - r0: rows.stop - r0, cols.start - c0: cols.stop - c0]

    with np.errstate(invalid="ignore", divide="ignore"):
        ls_mean = np.where(sums[1] > 0, sums[0] / sums[1], np.nan).astype(np.float32)

    profile = dict(driver="GTiff", height=height, width=width, count=1, dtype="float32", nodata=np.nan,
                   crs="EPSG:4326", transform=from_origin(west, north, FIVE_MIN_DEG, FIVE_MIN_DEG), compress="deflate")
    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_fp, "w", **profile) as dst:
        dst.write(ls_mean, 1)
    return ls_mean
