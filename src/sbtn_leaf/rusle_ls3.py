# RUSLE LS method LS3: Sun et al. (2024) LS-WPC / CSLE, computed from MERIT DEM tiles
'''
LS3 reproduces the algorithm of the DS-LS-GS1 global LS dataset:
Sun, Y., Zhang, H., Yang, Q. et al. (2024). A new high-resolution global topographic factor dataset calculated
based on SRTM. Scientific Data 11, 101 (RUSLE/papers/LS_2026.pdf). Settings are the LS-TOOL defaults (paper Fig. 13).

Per buffered DEM tile:
1. Cell size on the lat/lon grid, without projection (LS-WPC, Eqs. 4-5): N-S Cx = R * res, E-W Cy = Cx * cos(lat).
2. Sink fill: priority-flood with a small epsilon (Barnes et al. 2014), so every cell drains (LS-TOOL "Sink fill").
3. D8 (Eq. 6): theta = max over the 8 neighbours of arctan((z_c - z_i) / g); the outflow goes to that neighbour.
   theta is floored at 0.1 deg ("the slope steepness of the grid cell was set to 0.1 at a slope steepness of 0").
4. Cell slope length (CSL) = g to the downslope neighbour.
5. Cumulative slope length (Eq. 7), "Max" mode, in upstream-to-downstream order:
   lambda_in = max lambda_out over the donors whose flow is not cut; lambda_out = lambda_in + CSL. A donor j -> i is
   cut when the slope decreases by more than a cut-off factor, (theta_j - theta_i) / theta_j > 0.7 if theta_j < 2.86
   deg (5%), else 0.5; or when i is a channel cell (D8 contributing area > 800,000 m2). Channel cells restart at
   lambda_in = 0; they are not masked.
6. L (Eq. 9) = (lambda_out^(m+1) - lambda_in^(m+1)) / ((lambda_out - lambda_in) * 22.13^m),
   m (Eq. 10) = 0.2 (theta <= 0.5 deg), 0.3 (< 1.5), 0.4 (< 3), 0.5 (>= 3).
7. S, CSLE (Eq. 8): 10.8 sin + 0.03 (< 5 deg), 16.8 sin - 0.50 (5-10 deg), 21.91 sin - 0.96 (>= 10 deg).
   The paper prints -0.05 in the middle branch; -0.50 is the continuous (Liu et al. / McCool) value.
There is no slope-length cap and no slope clamp, as in the paper.

Differences from DS-LS-GS1: MERIT DEM at 3" instead of denoised SRTM at 1" (the paper's Suppl. Table 1 reports
higher LS on MERIT), and a 0.1-degree tile buffer instead of 1 degree (the paper's buffer test, Fig. 5, finds edge
effects gone by about 9 km at 30 m).

Only the DEM is needed (flow is routed here), so any MERIT DEM tar works without its upa tar. Output is the same
4-band sums file as rusle_ls.process_archive (LS_sum, n, LSw_sum, w_sum), so rusle_ls.mosaic_ls mosaics it.
'''
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from rasterio.transform import from_origin
from rasterio.windows import Window
from tqdm.auto import tqdm

from sbtn_leaf.rusle_ls import (
    BBox, EARTH_RADIUS_M, FIVE_MIN_DEG, LS_BANDS, MAX_SUM_BYTES, PathLike, UNIT_PLOT_LENGTH_M,
    _TileSet, _grid_index, _read_float, aggregate_block, write_sums,
)

try:
    from numba import njit
except ImportError:                                               # pragma: no cover
    njit = None

## Constants (LS-TOOL defaults, Sun et al. 2024 Fig. 13)
CHANNEL_AREA_M2 = 800_000.0          # contributing area that starts a channel, m2
CUTOFF_FACTORS = (0.7, 0.5)          # slope-decrease cut-off below / at-or-above the split slope
CUTOFF_SPLIT_DEG = 2.86              # 5% slope
MIN_SLOPE_DEG = 0.1                  # slope floor
PAPER_EARTH_RADIUS_M = 6_371_000.0   # R used for Cx in the paper (Eq. 4)
FILL_EPS_M = 1e-6                    # priority-flood epsilon, m
DEFAULT_BUFFER_PX = 120              # 0.1 degree at 3"

# D8 neighbours (row offset, column offset): E, SE, S, SW, W, NW, N, NE
_DR = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
_DC = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)


def _require_numba():
    if njit is None:
        raise ImportError("LS3 needs numba (pip install numba)")


# KERNELS (compiled on first use)
_KERNELS = {}


def _kernels():
    """Compile the numba kernels once and return them."""
    if _KERNELS:
        return _KERNELS
    _require_numba()
    import heapq

    @njit(cache=True)
    def priority_flood(z, eps):
        """Depression-filled copy of z (Barnes et al. 2014, priority-flood + epsilon). NaN cells stay NaN and act as
        outlets, as does the array border."""
        nr, nc = z.shape
        zf = z.copy()
        closed = np.zeros((nr, nc), dtype=np.bool_)
        heap = [(0.0, np.int64(0))]
        heap.pop()
        for r in range(nr):
            for c in range(nc):
                if not np.isfinite(z[r, c]):
                    closed[r, c] = True
                    continue
                seed = r == 0 or c == 0 or r == nr - 1 or c == nc - 1
                if not seed:
                    for k in range(8):
                        if not np.isfinite(z[r + _DR[k], c + _DC[k]]):
                            seed = True
                            break
                if seed:
                    closed[r, c] = True
                    heapq.heappush(heap, (zf[r, c], np.int64(r * nc + c)))
        while len(heap) > 0:
            zc, idx = heapq.heappop(heap)
            r, c = idx // nc, idx % nc
            for k in range(8):
                rr, cc = r + _DR[k], c + _DC[k]
                if rr < 0 or cc < 0 or rr >= nr or cc >= nc or closed[rr, cc]:
                    continue
                closed[rr, cc] = True
                if zf[rr, cc] <= zc:
                    zf[rr, cc] = zc + eps
                heapq.heappush(heap, (zf[rr, cc], np.int64(rr * nc + cc)))
        return zf

    @njit(cache=True)
    def d8(zf, dy, dx_rows, min_slope_rad):
        """Receiver (flat index, -1 for outlets), slope theta (rad) and cell slope length (m) per cell."""
        nr, nc = zf.shape
        recv = np.full(nr * nc, -1, dtype=np.int64)
        theta = np.full((nr, nc), np.nan, dtype=np.float64)
        csl = np.full((nr, nc), np.nan, dtype=np.float64)
        for r in range(nr):
            dx = dx_rows[r]
            diag = math.sqrt(dx * dx + dy * dy)
            for c in range(nc):
                zc = zf[r, c]
                if not np.isfinite(zc):
                    continue
                best, bk = -1e300, -1
                for k in range(8):
                    rr, cc = r + _DR[k], c + _DC[k]
                    if rr < 0 or cc < 0 or rr >= nr or cc >= nc:
                        continue
                    zn = zf[rr, cc]
                    if not np.isfinite(zn):
                        continue
                    g = dx if _DR[k] == 0 else (dy if _DC[k] == 0 else diag)
                    s = (zc - zn) / g
                    if s > best:
                        best, bk = s, k
                t = math.atan(best) if best > 0 else 0.0
                theta[r, c] = max(t, min_slope_rad)
                if bk >= 0 and best > 0:
                    recv[r * nc + c] = (r + _DR[bk]) * nc + (c + _DC[bk])
                    csl[r, c] = dx if _DR[bk] == 0 else (dy if _DC[bk] == 0 else diag)
                else:
                    csl[r, c] = math.sqrt(dx * dy)                 # outlet: nominal cell size
        return recv, theta, csl

    @njit(cache=True)
    def slope_length(recv, theta, csl, cell_area, channel_area, f_low, f_high, split_rad):
        """lambda_in, lambda_out (m) and D8 contributing area (m2), processing cells in topological order."""
        n = recv.size
        th = theta.ravel()
        cs = csl.ravel()
        ca = cell_area.ravel()
        indeg = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if recv[i] >= 0:
                indeg[recv[i]] += 1
        order = np.empty(n, dtype=np.int64)
        head, tail = 0, 0
        for i in range(n):
            if indeg[i] == 0 and np.isfinite(th[i]):
                order[tail] = i
                tail += 1
        area = np.where(np.isfinite(ca), ca, 0.0)
        lam_in = np.zeros(n, dtype=np.float64)
        lam_out = np.full(n, np.nan, dtype=np.float64)
        while head < tail:                                         # Kahn's algorithm; donors before receivers
            i = order[head]
            head += 1
            if area[i] > channel_area:                             # channel cell: the slope length restarts
                lam_in[i] = 0.0
            lam_out[i] = lam_in[i] + cs[i]
            j = recv[i]
            if j >= 0:
                area[j] += area[i]
                ti, tj = th[i], th[j]
                f = f_low if ti < split_rad else f_high
                if not ((ti - tj) / ti > f) and lam_out[i] > lam_in[j]:   # slope cut-off ends the slope length
                    lam_in[j] = lam_out[i]
                indeg[j] -= 1
                if indeg[j] == 0:
                    order[tail] = j
                    tail += 1
        return lam_in.reshape(theta.shape), lam_out.reshape(theta.shape), area.reshape(theta.shape)

    _KERNELS.update(priority_flood=priority_flood, d8=d8, slope_length=slope_length)
    return _KERNELS


# LS FORMULAS
def csle_s(theta_rad: np.ndarray) -> np.ndarray:
    """S factor of the CSLE (Sun et al. 2024 Eq. 8; middle branch -0.50, see the module docstring)."""
    deg = np.degrees(theta_rad)
    s = np.sin(theta_rad)
    return np.where(deg < 5.0, 10.8 * s + 0.03, np.where(deg < 10.0, 16.8 * s - 0.50, 21.91 * s - 0.96))


def ls3_m(theta_rad: np.ndarray) -> np.ndarray:
    """Slope-length exponent m (Sun et al. 2024 Eq. 10)."""
    deg = np.degrees(theta_rad)
    return np.where(deg <= 0.5, 0.2, np.where(deg < 1.5, 0.3, np.where(deg < 3.0, 0.4, 0.5)))


def segment_l(lam_in: np.ndarray, lam_out: np.ndarray, m: np.ndarray) -> np.ndarray:
    """L of a slope segment (Sun et al. 2024 Eq. 9); (lambda_out / 22.13)^m where the segment has no length."""
    with np.errstate(invalid="ignore", divide="ignore"):
        d = lam_out - lam_in
        seg = (lam_out ** (m + 1) - lam_in ** (m + 1)) / (d * UNIT_PLOT_LENGTH_M ** m)
        return np.where(d > 0, seg, (lam_out / UNIT_PLOT_LENGTH_M) ** m)


def ls3_from_dem(
    z: np.ndarray,
    lat_deg: Optional[np.ndarray] = None,
    res_deg: Optional[float] = None,
    dx_m: Optional[float] = None,
    dy_m: Optional[float] = None,
    channel_area_m2: float = CHANNEL_AREA_M2,
    cutoff: Tuple[float, float] = CUTOFF_FACTORS,
    cutoff_split_deg: float = CUTOFF_SPLIT_DEG,
    min_slope_deg: float = MIN_SLOPE_DEG,
    earth_radius_m: float = PAPER_EARTH_RADIUS_M,
    fill_eps_m: float = FILL_EPS_M,
    return_parts: bool = False,
):
    """LS3 for one DEM array (rows north -> south; NaN where missing).

    Geographic grid: give lat_deg (row centres) and res_deg. Metric grid (e.g. a synthetic test surface): give
    dx_m and dy_m instead. return_parts=True also returns a dict with theta (rad), lam_in, lam_out (m), area (m2), L
    and S.
    """
    k = _kernels()
    z = np.asarray(z, dtype=np.float64)
    nr = z.shape[0]
    if dx_m is not None and dy_m is not None:
        dy = float(dy_m)
        dx_rows = np.full(nr, float(dx_m))
    elif lat_deg is not None and res_deg is not None:
        dy = earth_radius_m * math.radians(res_deg)                # Cx in the paper (N-S)
        dx_rows = dy * np.cos(np.radians(np.asarray(lat_deg, dtype=np.float64)))   # Cy (E-W)
    else:
        raise ValueError("give either (lat_deg, res_deg) or (dx_m, dy_m)")
    zf = k["priority_flood"](z, fill_eps_m)
    recv, theta, csl = k["d8"](zf, dy, dx_rows, math.radians(min_slope_deg))
    cell_area = np.where(np.isfinite(z), dy * dx_rows[:, None], np.nan)
    lam_in, lam_out, area = k["slope_length"](recv, theta, csl, cell_area, float(channel_area_m2),
                                              float(cutoff[0]), float(cutoff[1]), math.radians(cutoff_split_deg))
    m = ls3_m(theta)
    L = segment_l(lam_in, lam_out, m)
    S = csle_s(theta)
    ls = L * S
    ls[~np.isfinite(z)] = np.nan
    if return_parts:
        return ls, dict(theta=theta, lam_in=lam_in, lam_out=lam_out, area=area, L=L, S=S)
    return ls


# TILE PROCESSING
def _read_buffered(ts: _TileSet, key, buf: int) -> np.ndarray:
    """Tile `key` with a `buf`-pixel border taken from the 8 neighbouring tiles (NaN where none)."""
    H, W = ts.height, ts.width
    out = np.full((H + 2 * buf, W + 2 * buf), np.nan)
    spans = {-1: (H - buf, H, 0), 0: (0, H, buf), 1: (0, buf, buf + H)}        # rows: src start, src stop, dst
    cspans = {-1: (W - buf, W, 0), 0: (0, W, buf), 1: (0, buf, buf + W)}
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            ds = ts.tiles.get(key) if (dr, dc) == (0, 0) else ts.neighbour(key, dc, dr)
            if ds is None:
                continue
            r0, r1, rd = spans[dr]
            c0, c1, cd = cspans[dc]
            out[rd: rd + r1 - r0, cd: cd + c1 - c0] = _read_float(ds, Window(c0, r0, c1 - c0, r1 - r0))
    return out


def process_archive_ls3(
    dem_tar: PathLike,
    out_fp: Optional[PathLike] = None,
    out_res_deg: float = FIVE_MIN_DEG,
    bbox: Optional[BBox] = None,
    only_tiles: Optional[Sequence[str]] = None,
    buffer_px: int = DEFAULT_BUFFER_PX,
    show_progress: bool = True,
    **ls3_kwargs,
):
    """LS3 sums and counts on the output grid for one MERIT DEM archive (the upa tar is not needed).

    Same output as rusle_ls.process_archive (bands LS_sum, n, LSw_sum, w_sum; the crop bands stay 0), so parts can
    be mosaicked with rusle_ls.mosaic_ls. only_tiles: DEM tile stems (e.g. "n45e005_dem"). ls3_kwargs go to
    ls3_from_dem (channel_area_m2, cutoff, ...). Returns (sums, transform).
    """
    dem = _TileSet(dem_tar)
    try:
        res, H, W = dem.res, dem.height, dem.width
        factor = round(out_res_deg / res)
        if factor < 1 or not math.isclose(factor * res, out_res_deg, rel_tol=1e-9) or H % factor or W % factor:
            raise ValueError(f"out_res_deg {out_res_deg} must be a whole multiple of {res} dividing the tile size")
        origins = {key: dem.origin_px(key) for key in dem.tiles}
        gx0, gy0 = min(o[0] for o in origins.values()), min(o[1] for o in origins.values())
        gx1, gy1 = max(o[0] for o in origins.values()) + W, max(o[1] for o in origins.values()) + H
        if bbox is not None:
            c0, c1, r0, r1 = _grid_index(bbox, out_res_deg)
            gx0, gx1 = max(gx0, c0 * factor), min(gx1, c1 * factor)
            gy0, gy1 = max(gy0, r0 * factor), min(gy1, r1 * factor)
        ncols, nrows = max(gx1 - gx0, 0) // factor, max(gy1 - gy0, 0) // factor
        transform = from_origin(-180.0 + gx0 * res, 90.0 - gy0 * res, out_res_deg, out_res_deg)
        if len(LS_BANDS) * nrows * ncols * 8 > MAX_SUM_BYTES:
            raise ValueError(f"output grid {nrows}x{ncols} is too large; pass a smaller bbox")
        sums = np.zeros((len(LS_BANDS), nrows, ncols))
        keys = list(dem.tiles)
        if only_tiles is not None:
            names = dem.by_name()
            missing = [t for t in only_tiles if t not in names]
            if missing:
                raise KeyError(f"tiles not in {dem_tar}: {missing}")
            keys = [names[t] for t in only_tiles]
        b = buffer_px
        for key in tqdm(keys, desc=Path(dem_tar).name, disable=not show_progress):
            tx0, ty0 = origins[key]
            wc0, wc1 = max(gx0, tx0) - tx0, min(gx1, tx0 + W) - tx0
            wr0, wr1 = max(gy0, ty0) - ty0, min(gy1, ty0 + H) - ty0
            if wc1 <= wc0 or wr1 <= wr0:
                continue
            z = _read_buffered(dem, key, b)
            if not np.isfinite(z[b:b + H, b:b + W]).any():
                continue
            tile_north = key[1] * res
            lat = tile_north + b * res - (np.arange(H + 2 * b) + 0.5) * res
            ls = ls3_from_dem(z, lat_deg=lat, res_deg=res, **ls3_kwargs)[b + wr0: b + wr1, b + wc0: b + wc1]
            blk = aggregate_block(ls, None, factor)
            orow, ocol = (ty0 + wr0 - gy0) // factor, (tx0 + wc0 - gx0) // factor
            for i in range(len(LS_BANDS)):
                sums[i, orow: orow + blk[i].shape[0], ocol: ocol + blk[i].shape[1]] += blk[i]
    finally:
        dem.close()
    if out_fp is not None and sums.size:
        write_sums(out_fp, sums, transform)
    return sums, transform
