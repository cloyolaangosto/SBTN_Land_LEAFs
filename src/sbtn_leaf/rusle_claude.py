# RUSLE script description
'''
This module contains GIS implementation of 3 parameters of the RUSLE model, or the Revised Universal Soil Loss Equation, to estimate annual soil erosion for different commodities. The equation is as simple as:
A = R*K*L*C*P

Here, R, K, and L parameters are implemented.
- R, which represents rainfall erosivity, depends on monthly precipitation. It's based on GloREDa, which publishes global monthly maps, are used as baseline. RCP scenarios are then overlaid to create new estimates.
- K, or soil erodibility, represents how easy the soil erodes, and of course depends on soil conditions. This is constant in time and is generated 1 time.
- L, or slope-length factor, is dimensionless and it's simply a factor that modifies erosion based on how long and steep the slope of the terrain is. This is also time-independent for human time horizons and is generated 1 time
'''
# SETUP
## Module imports
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm.auto import tqdm

from sbtn_leaf.paths import data_path

## Data Import

## Other setup
# SoilGrids 2.0 depth intervals (cm) used for the 0-30 cm topsoil, with their thickness as weight
SOILGRIDS_DEPTHS: Dict[str, float] = {"0-5cm": 5.0, "5-15cm": 10.0, "15-30cm": 15.0}
SOILGRIDS_NODATA = -32768

# USDA texture classes. 0 is reserved for nodata
USDA_TEXTURE_CLASSES: Dict[int, str] = {
    1: "sand",
    2: "loamy sand",
    3: "sandy loam",
    4: "loam",
    5: "silt loam",
    6: "silt",
    7: "sandy clay loam",
    8: "clay loam",
    9: "silty clay loam",
    10: "sandy clay",
    11: "silty clay",
    12: "clay",
}

# Wischmeier & Smith (1978) structure class s
# 1 very fine granular, 2 fine granular, 3 medium/coarse granular, 4 blocky, platy or massive
# Panagos et al. (2014) took s from the ESDB pedotransfer rules (FAO soil name + texture), which are not available
# globally. GloSEM derives s from texture (Borrelli et al. 2017), but the exact table is not published in the papers we
# have, so this texture lookup is our own assumption. The alternative is the constant s = 2 ("fine granular topsoil",
# AH 537 first approximation and the dominant value in Panagos et al. 2014).
STRUCTURE_DEFAULT = 2
STRUCTURE_BY_TEXTURE: Dict[int, int] = {
    1: 1, 2: 1,               # sand, loamy sand
    3: 2, 4: 2, 5: 2, 6: 2,   # sandy loam, loam, silt loam, silt
    7: 3, 8: 3, 9: 3,         # sandy clay loam, clay loam, silty clay loam
    10: 4, 11: 4, 12: 4,      # sandy clay, silty clay, clay
}

# Wischmeier & Smith (1978) permeability class p, by texture (Panagos et al. 2014, Table 2; USDA NSSH 430)
# 1 rapid, 2 moderate to rapid, 3 moderate, 4 slow to moderate, 5 slow, 6 very slow
PERMEABILITY_BY_TEXTURE: Dict[int, int] = {
    1: 1,                     # sand
    2: 2, 3: 2,               # loamy sand, sandy loam
    4: 3, 5: 3, 6: 3,         # loam, silt loam, silt
    7: 4, 8: 4,               # sandy clay loam, clay loam
    9: 5, 10: 5,              # silty clay loam, sandy clay
    11: 6, 12: 6,             # silty clay, clay
}

# Saturated hydraulic conductivity of the fine earth Kf (mm/h) by texture, Rawls et al. (1982).
# Silt is not in Rawls et al. (1982); it takes the silt loam value
KSAT_BY_TEXTURE: Dict[int, float] = {
    1: 210.0, 2: 61.1, 3: 25.9, 4: 13.2, 5: 6.8, 6: 6.8,
    7: 4.3, 8: 2.3, 9: 1.5, 10: 1.2, 11: 0.9, 12: 0.6,
}
# Lower Ksat bound (mm/h) of permeability classes 1-5 (Panagos et al. 2014, Table 2); below 1.0 is class 6
PERMEABILITY_KSAT_BREAKS = (61.0, 20.3, 5.1, 2.0, 1.0)

# Very fine sand (0.05-0.1 mm) as a share of total sand (0.05-2 mm). Not in SoilGrids; 20% follows Panagos et al. (2014).
# Their sensitivity test: 33.3% raises K by 11%
VFS_SAND_FRACTION = 0.2

# K nomograph constants
K_SI_CONVERSION = 0.1317        # US customary K -> t ha h ha-1 MJ-1 mm-1 (Foster et al. 1981)
OM_MAX = 4.0                    # Nomograph OM upper limit, % (Wischmeier et al. 1971; Panagos et al. 2014; Yang et al. 2024)
RUSLE2_SILT_VFS_BREAK = 68.0    # RUSLE2 texture term changes above 68% silt + vfs (Renard & Dabney 2013)
# Yang et al. (2024) global correction against 144 measured-K sites: K_corr = a * K_RUSLE2 + b (R2 = 0.636, NSE = 0.64)
YANG_RUSLE2_CORRECTION = (2.1311, -0.0359)


# FUNCTIONS
## R functions

##################################
######### MY CODE ################
##################################
"""
So I've found 4 methods to estimate K. Currently, the best method, as defined by Yang et al. (2024) require data that I do not have, which is the permeability class and the soil structure class.
So for now, we're starting with K-EPIC, and then checking if we can estimate the other values, which seems to be the most cryptic in all publications so far.
"""



##################################
######### CLAUDE CODE ############
##################################
## K Functions
def calculate_K(clay_fp, fines_fp, sand_fp, oc_fp):



    K_fp = np.zeros_like(clay_fp)
    return K_fp

def _calculate_M(silt, vfs, clay):
    """Nomograph particle-size parameter M = (silt % + vfs %) * (100 - clay %)."""
    return (silt + vfs) * (100.0 - clay)

def _calculate_OM(soc, om_max: Optional[float] = OM_MAX):
    """Organic matter % from SoilGrids soc (dg/kg): SOC % = soc / 100, OM = 1.724 * SOC. Capped at om_max (None = no cap)."""
    om = 1.724 * np.asarray(soc, dtype=np.float32) / 100.0
    return np.minimum(om, om_max) if om_max is not None else om

def k_factor_rusle2(
    sand, silt, clay, om, s, p,
    vfs: Optional[np.ndarray] = None,
    om_max: Optional[float] = OM_MAX,
    correct: bool = False,
) -> np.ndarray:
    """
    RUSLE2 soil erodibility K (t ha h ha-1 MJ-1 mm-1), as applied globally by Yang et al. (2024), after Renard & Dabney (2013):

        K   = 0.1317 * (Kt * Ko + Ks + Kp) / 100
        Kt  = Ktb                                  if silt + vfs <= 68 %
              Ktb - 0.67 * (Ktb - Kt68) ** 0.82    otherwise
        Ktb = 2.1e-4 * ((silt + vfs) * (100 - clay)) ** 1.14
        Kt68 = 2.1e-4 * (68 * (100 - clay)) ** 1.14
        Ko  = 12 - OM,  Ks = 3.25 * (s - 2),  Kp = 2.5 * (p - 3)

    Below 68 % silt + vfs this is the USLE nomograph equation. Above it, the texture term flattens instead of
    the silt being capped at 70 %. Yang et al. print Ks = 3.25 * (2 - s); we keep the nomograph sign (s - 2), since
    more blocky/massive structure must raise K (AH 537, Fig. 3).

    sand, silt, clay in any consistent unit (normalised to %); om in %; s and p class codes (0 = nodata).
    vfs in % of fine earth; if None it is estimated with the RUSLE2 equation (_calculate_vfs, method="rusle2").
    correct=True applies the Yang et al. (2024) measured-K correction (floored at 0).
    Returns float32, NaN = nodata.
    """
    sa, si, cl, valid = _normalise_texture(sand, silt, clay)
    if vfs is None:
        vfs = _calculate_vfs(sa, si, cl, method="rusle2")
    om = np.asarray(om, dtype=np.float32)
    if om_max is not None:
        om = np.minimum(om, om_max)
    s = np.asarray(s)
    p = np.asarray(p)
    valid &= np.isfinite(vfs) & np.isfinite(om) & (s > 0) & (p > 0)

    with np.errstate(invalid="ignore"):
        ktb = 2.1e-4 * _calculate_M(si, vfs, cl) ** 1.14
        kt68 = 2.1e-4 * (RUSLE2_SILT_VFS_BREAK * (100.0 - cl)) ** 1.14
        kt = np.where(si + vfs <= RUSLE2_SILT_VFS_BREAK, ktb, ktb - 0.67 * np.clip(ktb - kt68, 0, None) ** 0.82)
        k = K_SI_CONVERSION * (kt * (12.0 - om) + 3.25 * (s - 2.0) + 2.5 * (p - 3.0)) / 100.0

    if correct:
        k = yang_rusle2_correction(k)
    return np.where(valid, k, np.nan).astype(np.float32)

def yang_rusle2_correction(k):
    """Yang et al. (2024) Eq. 26: K_corr = 2.1311 * K_RUSLE2 - 0.0359, floored at 0 (negative below K ~ 0.0168)."""
    a, b = YANG_RUSLE2_CORRECTION
    return np.maximum(a * np.asarray(k) + b, 0.0)

def _lookup(classes: np.ndarray, table: Dict[int, int]) -> np.ndarray:
    """Map integer class codes through ``table``. Codes not in the table (e.g. 0 = nodata) map to 0."""
    lut = np.zeros(max(max(table), int(classes.max(initial=0))) + 1, dtype=np.uint8)
    for k, v in table.items():
        lut[k] = v
    return lut[classes]

def _normalise_texture(sand, silt, clay):
    """
    Rescale sand, silt and clay so they sum to 100 %. Inputs in any consistent unit (SoilGrids: g/kg).
    Returns float32 percentages and a valid-data mask (finite and sum > 0).
    """
    sand = np.asarray(sand, dtype=np.float32)
    silt = np.asarray(silt, dtype=np.float32)
    clay = np.asarray(clay, dtype=np.float32)
    total = sand + silt + clay
    valid = np.isfinite(total) & (total > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(valid, 100.0 / total, np.nan).astype(np.float32)
    return sand * scale, silt * scale, clay * scale, valid

def usda_texture_class(sand, silt, clay) -> np.ndarray:
    """
    USDA texture triangle classification (codes in USDA_TEXTURE_CLASSES, 0 = nodata).
    sand, silt, clay can be in g/kg or %, they are normalised to sum 100 % first.
    """
    sa, si, cl, valid = _normalise_texture(sand, silt, clay)

    # Order matters: np.select takes the first matching condition
    conditions = [
        (si + 1.5 * cl) < 15,                                                       # sand
        ((si + 1.5 * cl) >= 15) & ((si + 2 * cl) < 30),                             # loamy sand
        ((cl >= 7) & (cl < 20) & (sa > 52) & ((si + 2 * cl) >= 30))
        | ((cl < 7) & (si < 50) & ((si + 2 * cl) >= 30)),                           # sandy loam
        (cl >= 7) & (cl < 27) & (si >= 28) & (si < 50) & (sa <= 52),                # loam
        ((si >= 50) & (cl >= 12) & (cl < 27)) | ((si >= 50) & (si < 80) & (cl < 12)),  # silt loam
        (si >= 80) & (cl < 12),                                                     # silt
        (cl >= 20) & (cl < 35) & (si < 28) & (sa > 45),                             # sandy clay loam
        (cl >= 27) & (cl < 40) & (sa > 20) & (sa <= 45),                            # clay loam
        (cl >= 27) & (cl < 40) & (sa <= 20),                                        # silty clay loam
        (cl >= 35) & (sa > 45),                                                     # sandy clay
        (cl >= 40) & (si >= 40),                                                    # silty clay
        (cl >= 40) & (sa <= 45) & (si < 40),                                        # clay
    ]
    texture = np.select(conditions, list(USDA_TEXTURE_CLASSES), default=0).astype(np.uint8)
    texture[~valid] = 0
    return texture

def _calculate_soil_structure(
    texture: np.ndarray,
    om: Optional[np.ndarray] = None,
    om_thresholds: Optional[Sequence[float]] = None,
    method: str = "texture",
) -> np.ndarray:
    """
    Soil structure class s (1-4) of the Wischmeier & Smith (1978) nomograph.

    method = "texture": lookup by USDA texture class (STRUCTURE_BY_TEXTURE, our assumption).
    method = "constant": s = STRUCTURE_DEFAULT (2, fine granular) everywhere with data.

    Optional organic matter adjustment (off by default, for sensitivity tests): with
    om_thresholds = (low, high) in OM %, s is raised one class (coarser / more massive) where OM < low
    and lowered one class (finer granular) where OM > high, clipped to 1-4.

    Returns uint8 array, 0 = nodata.
    """
    if method == "texture":
        s = _lookup(texture, STRUCTURE_BY_TEXTURE)
    elif method == "constant":
        s = np.where(texture > 0, STRUCTURE_DEFAULT, 0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown structure method: {method}")
    if om is not None and om_thresholds is not None:
        low, high = om_thresholds
        valid = s > 0
        shift = np.where(om < low, 1, 0) - np.where(om > high, 1, 0)
        s = np.where(valid, np.clip(s.astype(np.int16) + shift, 1, 4), 0).astype(np.uint8)
    return s

def _calculate_perm_class(
    texture: np.ndarray,
    coarse_fragments: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Permeability class p (1-6) of the Wischmeier & Smith (1978) nomograph, from USDA texture class.

    If coarse_fragments (vol %, > 2 mm) is given, p is calibrated as in Panagos et al. (2014): the fine-earth
    conductivity Kf (Rawls et al. 1982) is reduced to Kb = Kf * (1 - Rw) (Brakensiek et al. 1986), and p moves to
    the slower class of Kb when it falls below the texture class range. Stones only slow permeability, so p never
    decreases. In Europe this moved the dominant p from 3 to 4 and raised K by ~15%.

    Returns uint8 array, 0 = nodata.
    """
    p = _lookup(texture, PERMEABILITY_BY_TEXTURE)
    if coarse_fragments is not None:
        kf = np.zeros(13, dtype=np.float32)
        for k, v in KSAT_BY_TEXTURE.items():
            kf[k] = v
        rw = np.clip(np.nan_to_num(coarse_fragments, nan=0.0), 0, 100) / 100.0
        kb = kf[texture] * (1.0 - rw)
        # Class of Kb: 1 + number of breaks Kb is not above
        p_kb = 1 + (kb[..., None] <= np.asarray(PERMEABILITY_KSAT_BREAKS, dtype=np.float32)).sum(axis=-1)
        p = np.where(p > 0, np.maximum(p, p_kb), 0).astype(np.uint8)
    return p

def _calculate_vfs(sand, silt, clay, fraction: float = VFS_SAND_FRACTION, method: str = "fraction") -> np.ndarray:
    """
    Very fine sand (0.05-0.1 mm), in % of the fine earth. Inputs are normalised to sum 100 % first.

    method = "fraction": vfs = fraction * sand (Panagos et al. 2014, fraction = 0.2).
    method = "rusle2":   vfs = (0.74 - 0.62 * sand / 100) * sand (RUSLE2; Yang et al. 2024, Eq. 7). This is
                         up to 74% of sand in fine soils, falling to 12% in pure sand.

    Returns float32, NaN = nodata. The USLE nomograph is only valid for silt + vfs <= 70 %.
    """
    sa, _, _, valid = _normalise_texture(sand, silt, clay)
    if method == "fraction":
        vfs = sa * fraction
    elif method == "rusle2":
        vfs = (0.74 - 0.62 * sa / 100.0) * sa
    else:
        raise ValueError(f"Unknown vfs method: {method}")
    return np.where(valid, vfs, np.nan).astype(np.float32)

def _depth_weighted_mean(layers: Sequence[np.ndarray], weights: Sequence[float], nodata=SOILGRIDS_NODATA) -> np.ndarray:
    """Thickness-weighted mean of several depth layers. A pixel is NaN if any layer is nodata."""
    stack = np.stack([np.asarray(a, dtype=np.float32) for a in layers])
    stack[stack == nodata] = np.nan
    w = np.asarray(weights, dtype=np.float32)[:, None, None]
    return (stack * w).sum(axis=0) / w.sum()

def build_structure_permeability_vfs(
    soilgrids_dir: Union[str, Path, None] = None,
    out_dir: Union[str, Path, None] = None,
    depths: Optional[Sequence[str]] = None,
    vfs_fraction: float = VFS_SAND_FRACTION,
    structure_method: str = "texture",
    om_thresholds: Optional[Sequence[float]] = None,
    use_coarse_fragments: bool = True,
    tile_size: int = 2048,
    suffix: str = "mean_1000",
) -> Dict[str, Path]:
    """
    Build the texture-derived inputs of the K1 nomograph from SoilGrids 2.0 rasters, on their native grid
    (1 km Homolosine for the 1000 m release). Processes tile by tile, so the global grid is never in memory.

    Inputs read from soilgrids_dir: {sand,silt,clay}_{depth}_{suffix}.tif (g/kg). Also cfvo (cm3/dm3) if
    use_coarse_fragments (Panagos et al. 2014 permeability calibration), and soc (dg/kg) if om_thresholds is set.
    Depths are thickness-weighted to 0-30 cm. structure_method is "texture" or "constant" (see _calculate_soil_structure).

    Outputs written to out_dir:
    - usda_texture_0-30cm.tif  (uint8, codes in USDA_TEXTURE_CLASSES, 0 = nodata)
    - K_structure_s_0-30cm.tif (uint8, 1-4, 0 = nodata)
    - K_permeability_p_0-30cm.tif (uint8, 1-6, 0 = nodata)
    - K_vfs_0-30cm.tif (float32, % of fine earth, NaN = nodata)

    Returns a dict of output paths.
    """
    soilgrids_dir = Path(soilgrids_dir) if soilgrids_dir else data_path("soilgrids_1000m")
    out_dir = Path(out_dir) if out_dir else data_path("soil_erosion", "K_1km")
    out_dir.mkdir(parents=True, exist_ok=True)
    depths = list(depths) if depths else list(SOILGRIDS_DEPTHS)
    weights = [SOILGRIDS_DEPTHS[d] for d in depths]

    props = ["sand", "silt", "clay"]
    if om_thresholds is not None:
        props.append("soc")
    if use_coarse_fragments:
        props.append("cfvo")

    in_paths = {p: [soilgrids_dir / f"{p}_{d}_{suffix}.tif" for d in depths] for p in props}
    missing = [str(f) for fs in in_paths.values() for f in fs if not f.exists()]
    if missing:
        raise FileNotFoundError("Missing SoilGrids inputs (still downloading?):\n" + "\n".join(missing))

    out_paths = {
        "texture": out_dir / "usda_texture_0-30cm.tif",
        "s": out_dir / "K_structure_s_0-30cm.tif",
        "p": out_dir / "K_permeability_p_0-30cm.tif",
        "vfs": out_dir / "K_vfs_0-30cm.tif",
    }

    srcs = {p: [rasterio.open(f) for f in fs] for p, fs in in_paths.items()}
    try:
        ref = srcs["sand"][0]
        base = ref.profile.copy()
        base.update(count=1, tiled=True, blockxsize=512, blockysize=512, compress="deflate", BIGTIFF="IF_SAFER")
        int_profile = {**base, "dtype": "uint8", "nodata": 0}
        vfs_profile = {**base, "dtype": "float32", "nodata": np.nan, "predictor": 3}

        dsts = {k: rasterio.open(v, "w", **(vfs_profile if k == "vfs" else int_profile)) for k, v in out_paths.items()}
        try:
            for win in tqdm(_tile_windows(ref.width, ref.height, tile_size), desc="s, p, vfs tiles"):
                vals = {p: _depth_weighted_mean([s.read(1, window=win) for s in ss], weights) for p, ss in srcs.items()}

                texture = usda_texture_class(vals["sand"], vals["silt"], vals["clay"])
                om = _calculate_OM(vals["soc"], om_max=None) if "soc" in vals else None
                cf = vals["cfvo"] / 10.0 if "cfvo" in vals else None          # cm3/dm3 -> vol %

                dsts["texture"].write(texture, 1, window=win)
                dsts["s"].write(_calculate_soil_structure(texture, om, om_thresholds, structure_method), 1, window=win)
                dsts["p"].write(_calculate_perm_class(texture, cf), 1, window=win)
                dsts["vfs"].write(_calculate_vfs(vals["sand"], vals["silt"], vals["clay"], vfs_fraction), 1, window=win)
        finally:
            for d in dsts.values():
                d.close()
    finally:
        for ss in srcs.values():
            for s in ss:
                s.close()

    return out_paths

def build_k_rusle2(
    soilgrids_dir: Union[str, Path, None] = None,
    k_inputs_dir: Union[str, Path, None] = None,
    out_dir: Union[str, Path, None] = None,
    depths: Optional[Sequence[str]] = None,
    vfs_method: str = "rusle2",
    vfs_fraction: float = VFS_SAND_FRACTION,
    om_max: Optional[float] = OM_MAX,
    tile_size: int = 2048,
    suffix: str = "mean_1000",
) -> Dict[str, Path]:
    """
    RUSLE2-K (Yang et al. 2024) on the native SoilGrids grid, tile by tile.

    Reads {sand,silt,clay,soc}_{depth}_{suffix}.tif from soilgrids_dir (depth-weighted to 0-30 cm) and the s and p
    rasters written by build_structure_permeability_vfs from k_inputs_dir (same grid). vfs is recomputed here with
    vfs_method ("rusle2" as in Yang et al., or "fraction" with vfs_fraction as in Panagos et al.).

    Outputs written to out_dir (float32, NaN = nodata, t ha h ha-1 MJ-1 mm-1):
    - K_rusle2_0-30cm.tif: uncorrected RUSLE2-K
    - K_rusle2_yang_corrected_0-30cm.tif: with the Yang et al. (2024) measured-K correction

    Returns a dict of output paths.
    """
    soilgrids_dir = Path(soilgrids_dir) if soilgrids_dir else data_path("soilgrids_1000m")
    k_inputs_dir = Path(k_inputs_dir) if k_inputs_dir else data_path("soil_erosion", "K_1km")
    out_dir = Path(out_dir) if out_dir else k_inputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    depths = list(depths) if depths else list(SOILGRIDS_DEPTHS)
    weights = [SOILGRIDS_DEPTHS[d] for d in depths]

    in_paths = {p: [soilgrids_dir / f"{p}_{d}_{suffix}.tif" for d in depths] for p in ["sand", "silt", "clay", "soc"]}
    class_paths = {"s": k_inputs_dir / "K_structure_s_0-30cm.tif", "p": k_inputs_dir / "K_permeability_p_0-30cm.tif"}
    missing = [str(f) for fs in in_paths.values() for f in fs if not f.exists()]
    missing += [str(f) for f in class_paths.values() if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing inputs (SoilGrids still downloading, or run build_structure_permeability_vfs first):\n"
            + "\n".join(missing)
        )

    out_paths = {
        "k": out_dir / "K_rusle2_0-30cm.tif",
        "k_corrected": out_dir / "K_rusle2_yang_corrected_0-30cm.tif",
    }

    srcs = {p: [rasterio.open(f) for f in fs] for p, fs in in_paths.items()}
    cls = {k: rasterio.open(v) for k, v in class_paths.items()}
    try:
        ref = srcs["sand"][0]
        if any(c.shape != ref.shape or c.transform != ref.transform for c in cls.values()):
            raise ValueError("s/p rasters are not on the SoilGrids grid; rebuild them from the same SoilGrids release")
        profile = ref.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan, tiled=True, blockxsize=512, blockysize=512,
                       compress="deflate", predictor=3, BIGTIFF="IF_SAFER")

        dsts = {k: rasterio.open(v, "w", **profile) for k, v in out_paths.items()}
        try:
            for win in tqdm(_tile_windows(ref.width, ref.height, tile_size), desc="RUSLE2-K tiles"):
                vals = {p: _depth_weighted_mean([s.read(1, window=win) for s in ss], weights) for p, ss in srcs.items()}
                s = cls["s"].read(1, window=win)
                p = cls["p"].read(1, window=win)
                vfs = _calculate_vfs(vals["sand"], vals["silt"], vals["clay"], vfs_fraction, vfs_method)
                om = _calculate_OM(vals["soc"], om_max)

                k = k_factor_rusle2(vals["sand"], vals["silt"], vals["clay"], om, s, p, vfs=vfs, om_max=om_max)
                dsts["k"].write(k, 1, window=win)
                dsts["k_corrected"].write(yang_rusle2_correction(k).astype(np.float32), 1, window=win)
        finally:
            for d in dsts.values():
                d.close()
    finally:
        for ss in srcs.values():
            for s in ss:
                s.close()
        for c in cls.values():
            c.close()

    return out_paths

def _tile_windows(width: int, height: int, tile_size: int):
    """Windows of tile_size x tile_size covering a width x height raster."""
    return [
        Window(c, r, min(tile_size, width - c), min(tile_size, height - r))
        for r in range(0, height, tile_size)
        for c in range(0, width, tile_size)
    ]

## L Functions
