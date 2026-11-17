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
from typing import Any, Dict, Optional, Sequence, Union
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm.auto import tqdm

from affine import Affine
from rasterio.crs import CRS

from sbtn_leaf.paths import data_path
PathLike = Union[str, Path]

## Data Import
CLAY_LIST = [data_path("soilgrids_1000m","clay_0-5cm_mean_1000.tif"),
             data_path("soilgrids_1000m","clay_5-15cm_mean_1000.tif"),
             data_path("soilgrids_1000m","clay_15-30cm_mean_1000.tif")
            ]

SAND_LIST = [data_path("soilgrids_1000m","sand_0-5cm_mean_1000.tif"),
             data_path("soilgrids_1000m","sand_5-15cm_mean_1000.tif"),
             data_path("soilgrids_1000m","sand_15-30cm_mean_1000.tif")
            ]

SILT_LIST = [data_path("soilgrids_1000m","silt_0-5cm_mean_1000.tif"),
             data_path("soilgrids_1000m","silt_5-15cm_mean_1000.tif"),
             data_path("soilgrids_1000m","silt_15-30cm_mean_1000.tif")
            ]

SOC_LIST = [data_path("soilgrids_1000m","soc_0-5cm_mean_1000.tif"),
             data_path("soilgrids_1000m","soc_5-15cm_mean_1000.tif"),
             data_path("soilgrids_1000m","soc_15-30cm_mean_1000.tif")
            ]

## Other setup
# SoilGrids 2.0 depth intervals (cm) used for the 0-30 cm topsoil, with their thickness as weight
SOILGRIDS_DEPTHS: Dict[str, float] = {"0-5cm": 5.0, "5-15cm": 10.0, "15-30cm": 15.0}
SOILGRIDS_NODATA = -32768

OM_MAX = 4.0                    # Nomograph OM upper limit, % (Wischmeier et al. 1971; Panagos et al. 2014; Yang et al. 2024)

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

NUMBER_TEXTURE_CLASSES: Dict[int, str] = {
    12: "sand",
    11: "loamy sand",
    10: "sandy loam",
    9: "loam",
    8: "silt loam",
    7: "silt",
    6: "sandy clay loam",
    5: "clay loam",
    4: "silty clay loam",
    3: "sandy clay",
    2: "silty clay",
    1: "clay",
}

# Permeability class per USDA texture class (Panagos et al. 2014, after Rawls et al. 1982 / USDA NEH)
# 1 = rapid, 2 = moderate to rapid, 3 = moderate, 4 = slow to moderate, 5 = slow, 6 = very slow
PERMEABILITY_CLASSES: Dict[str, int] = {
    "sand": 1,
    "loamy sand": 2,
    "sandy loam": 2,
    "loam": 3,
    "silt loam": 3,
    "silt": 3,
    "sandy clay loam": 4,
    "clay loam": 4,
    "silty clay loam": 5,
    "sandy clay": 5,
    "silty clay": 6,
    "clay": 6,
}

# Lookup arrays indexed by USDA texture code (index 0 = nodata)
PERMEABILITY_BY_TEXTURE = np.array(
    [0] + [PERMEABILITY_CLASSES[USDA_TEXTURE_CLASSES[i]] for i in range(1, 13)], dtype=np.uint8
)
_NUMBER_BY_NAME = {name: num for num, name in NUMBER_TEXTURE_CLASSES.items()}
NUMBER_BY_TEXTURE = np.array(
    [0] + [_NUMBER_BY_NAME[USDA_TEXTURE_CLASSES[i]] for i in range(1, 13)], dtype=np.uint8
)

# FAO texture classes used in the European Soil Database (Jones, Spoor & Thomasson 2003, Table 1). 0 = nodata
FAO_TEXTURE_CLASSES: Dict[int, str] = {
    1: "coarse",        # clay < 18% and sand > 65%
    2: "medium",        # clay < 35% and sand > 15%; clay >= 18% if sand > 65%
    3: "medium fine",   # clay < 35% and sand < 15%
    4: "fine",          # clay 35-60%
    5: "very fine",     # clay > 60%
}
# Structure class proposed by Panagos 2014
STRUCTURE_BY_FAO_TEXTURE: Dict[int, int] = {
    1: 3,   # coarse -> very fine granular
    2: 3,   # medium -> fine granular
    3: 3,   # medium fine -> fine granular
    4: 2,   # fine -> blocky
    5: 1,   # very fine -> blocky
}
STRUCTURE_BY_FAO_TEXTURE_LUT = np.array([0] + [STRUCTURE_BY_FAO_TEXTURE[i] for i in range(1, 6)], dtype=np.uint8)

# Soil grids data class
@dataclass(frozen=True)
class SoilGridsData:
    """
    Soil Grids data layers outputs
    """
    soil_data: np.ndarray
    soil_meta: Dict[str, Any]
    soil_crs: CRS 
    soil_transform: Affine
    soil_width: int
    soil_height: int
    soil_nodata: float


####################################
######### MY K CODE ################
####################################
"""
So I've found 4 methods to estimate K. Currently, the best method, as defined by Yang et al. (2024) require data that I do not have, which is the permeability class and the soil structure class.
So for now, we're starting with K-EPIC, and then checking if we can estimate the other values, which seems to be the most cryptic in all publications so far.
"""

def calculate_K_Epic(output_fp_k_epic: str, output_fp_k_epic_corr, output_nd = np.nan, clay_fp_list = CLAY_LIST, silt_fp_list = SILT_LIST, sand_fp_list = SAND_LIST, soc_fp_list = SOC_LIST, raster_band = 1):
    print('Running K EPIC calculations...')
    
    # First, get all data opened and calculated weights
    print('     Loading data')
    clay_w_data = _calculate_weighted_soil_layers(clay_fp_list, nodata=output_nd)
    silt_w_data = _calculate_weighted_soil_layers(silt_fp_list, nodata=output_nd)
    sand_w_data = _calculate_weighted_soil_layers(sand_fp_list, nodata=output_nd)
    soc_w_data = _calculate_weighted_soil_layers(soc_fp_list, nodata=output_nd)

    # Transforming units into %
    clay = clay_w_data.soil_data / 10  # Transformation obtained from soilgrids FAQ page (https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html)
    silt = silt_w_data.soil_data / 10
    sand = sand_w_data.soil_data / 10
    soc = soc_w_data.soil_data / (10 * 10)

    print('     Calculating K Epic')
    # Nodata value
    nd = clay_w_data.soil_nodata

    # validity mask across all inputs
    valid = np.ones(clay.shape, dtype=bool)
    for arr in (clay, silt, sand, soc):
        valid &= (arr != nd) & ~np.isnan(arr)
    valid &= (clay + silt) > 0

    # initializing components of K_EPIC calculations
    alpha, beta, gamma, epsilon, k_epic, k_epic_corrected = (
        np.full(clay.shape, nd, dtype=np.float32) for _ in range(6)
    )

    # Working on valid pixels only:
    cy, st, sd, oc = clay[valid], silt[valid], sand[valid], soc[valid]
    sm = 1-sd/100

    # Calculating 
    alpha[valid] = 0.2 + 0.3 * np.exp(-0.0256 * sd * (1 - st / 100))
    beta[valid] = (st / (cy + st)) ** 0.3
    gamma[valid] = (1-(0.25*oc)/(oc+np.exp(3.72 - 2.95*oc)))
    epsilon[valid] = (1-(0.7*sm)/(sm+np.exp(-5.51+22.9*sm)))

    # Finally multiplying everything for K Epic
    k_epic[valid] = alpha[valid] * beta[valid] * gamma[valid] * epsilon[valid]

    # Transforming units (Yang et al. 2024, eq. 11 includes the 0.1317 factor)
    k_epic[valid] = k_epic[valid] * 0.1317

    # Applying Yang et al 2024 correction (eq. 27, on K in SI units)
    k_epic_corrected[valid] = (1.8722 * k_epic[valid]) - 0.05

    # Writing both files
    print('     Writing...')
    output_meta = clay_w_data.soil_meta.copy()
    output_meta.update(dtype="float32", count=1, nodata=clay_w_data.soil_nodata, compress="lzw")
    
    with rasterio.open(output_fp_k_epic, "w", **output_meta) as dst:
        dst.write(k_epic.astype("float32"), 1)
    print(f'     K EPIC saved into {output_fp_k_epic}')

    with rasterio.open(output_fp_k_epic_corr, "w", **output_meta) as dst:
        dst.write(k_epic_corrected.astype("float32"), 1)
    print(f'     K EPIC corrected saved into {output_fp_k_epic_corr}')

    print('All done!')

def calculate_K_RUSLE_2(output_fp_k: str, output_nd = np.nan, clay_fp_list = CLAY_LIST, silt_fp_list = SILT_LIST, sand_fp_list = SAND_LIST, soc_fp_list = SOC_LIST, raster_band = 1):
    """RUSLE2 K factor (Yang et al. 2024, Eqs. 3-10) on the SoilGrids 1 km grid, in t ha h ha-1 MJ-1 mm-1.

    The structure class s comes from the FAO texture class of the SoilGrids texture (Jones et al. 2003, Table 1;
    STRUCTURE_BY_FAO_TEXTURE).
    """
    print("Calculating global K RUSLE2")
    
    # Loading rasters
    clay, silt, sand, soc, soil_nd, soil_meta = _load_soil_data_as_percentage(clay_fp_list, silt_fp_list, sand_fp_list, soc_fp_list , raster_band, output_nd)

    print("     Calculating K")
    # Validity mask across all inputs (works for NaN and numeric nodata)
    valid = np.ones(sand.shape, dtype=bool)
    for arr in (clay, silt, sand, soc):
        valid &= np.isfinite(arr) & (arr != soil_nd)

    # Normalizing sand/silt/clay to 100% so K_t and the texture class use the same fractions
    sand, silt, clay, valid_norm = _normalize_sand_silt_clay(sand, silt, clay)
    valid &= valid_norm

    # Structure class s (from the FAO texture class, Jones et al. 2003 Table 1) and permeability class.
    # Cast from uint8 to float before subtracting, otherwise negative results wrap around.
    (texture, s, p) = _determine_usda_class_number_permeability(sand, silt, clay)
    valid &= texture > 0  # pixels that fall in no USDA texture class
    s = s.astype(np.float32)
    p = p.astype(np.float32)

    # Working on valid pixels only
    cy, st, sd, oc = clay[valid], silt[valid], sand[valid], soc[valid]
    s_v, p_v = s[valid], p[valid]

    # Very fine sand (vfs)
    vfs = (0.74 - 0.62 * sd / 100) * sd

    # Texture subfactor (K_t)
    m = st + vfs
    k_t68 = 2.1e-4 * (68 * (100 - cy)) ** 1.14
    k_tb = 2.1e-4 * (m * (100 - cy)) ** 1.14
    # RUSLE2 adjustment for silt + vfs >= 68 (exponent 0.82 per RUSLE2 science documentation; verify against source)
    K_t = np.where(m < 68, k_tb, k_tb - 0.67 * np.clip(k_tb - k_t68, 0, None) ** 0.82)

    # Organic matter subfactor (K_o)
    OM = _calculate_OM(oc)
    K_o = 12 - OM

    # Soil structure subfactor (K_s)
    K_s = 3.25 * (2 - s_v)

    # Soil profile permeability subfactor (K_p)
    K_p = 2.5 * (p_v - 3)

    # Finally calculating K (nomograph value / 100, converted to SI with 0.1317)
    K = np.full(sand.shape, output_nd, dtype=np.float32)
    K[valid] = 0.1317 * (K_t * K_o + K_s + K_p) / 100

    # Writing the output
    print('     Writing...')
    output_meta = soil_meta.copy()
    output_meta.update(dtype="float32", count=1, nodata=output_nd, compress="lzw")
        
    with rasterio.open(output_fp_k, "w", **output_meta) as dst:
        dst.write(K.astype("float32"), 1)
    print(f'     K RUSLE2 saved into {output_fp_k}')

def _load_soil_data_as_percentage(clay_fp_list = CLAY_LIST, silt_fp_list = SILT_LIST, sand_fp_list = SAND_LIST, soc_fp_list = SOC_LIST, raster_band = 1, output_nd = np.nan):
    print('     Loading data')
    clay_w_data = _calculate_weighted_soil_layers(clay_fp_list, raster_band=raster_band, nodata=output_nd)
    silt_w_data = _calculate_weighted_soil_layers(silt_fp_list, raster_band=raster_band, nodata=output_nd)
    sand_w_data = _calculate_weighted_soil_layers(sand_fp_list, raster_band=raster_band, nodata=output_nd)
    soc_w_data = _calculate_weighted_soil_layers(soc_fp_list, raster_band=raster_band, nodata=output_nd)

    # Transforming units into %
    clay = clay_w_data.soil_data / 10  # Transformation obtained from soilgrids FAQ page (https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html)
    silt = silt_w_data.soil_data / 10
    sand = sand_w_data.soil_data / 10
    soc = soc_w_data.soil_data / (10 * 10)

    soil_nd = clay_w_data.soil_nodata
    soil_meta = clay_w_data.soil_meta

    return (clay, silt, sand, soc, soil_nd, soil_meta)

def _calculate_weighted_soil_layers(soil_data_fp_list, raster_band = 1, nodata=np.nan) -> SoilGridsData:
    with (
        rasterio.open(soil_data_fp_list[0]) as src_data5,
        rasterio.open(soil_data_fp_list[1]) as src_data15,
        rasterio.open(soil_data_fp_list[2]) as src_data30,
    ):
        data_5 = src_data5.read(raster_band, out_dtype="float32")
        meta_5 = src_data5.meta.copy()         # Copy metadata for potential output.
        crs_5 = src_data5.crs             # Get the coordinate reference system of the first raster.
        transform_5 = src_data5.transform
        width_5 = src_data5.width
        height_5 = src_data5.height
        nodata_5 = src_data5.nodata

        data_15 = src_data15.read(raster_band, out_dtype="float32")
        crs_15 = src_data15.crs             # Get the coordinate reference system of the first raster.
        transform_15 = src_data15.transform
        width_15 = src_data15.width
        height_15 = src_data15.height
        nodata_15 = src_data15.nodata

        data_30 = src_data30.read(raster_band, out_dtype="float32")
        crs_30 = src_data30.crs             # Get the coordinate reference system of the first raster.
        transform_30 = src_data30.transform
        width_30 = src_data30.width
        height_30 = src_data30.height
        nodata_30 = src_data30.nodata

        # Checks that all data is in the same format
        if (crs_5 != crs_15) or (crs_15 != crs_30) or (crs_5 != crs_30):
            raise ValueError("One of the data sources has different CRS")

        if (transform_5 != transform_15) or (transform_15 != transform_30) or (transform_5 != transform_30):
            raise ValueError("One of the data sources has different transform")

        if (width_5 != width_15) or (width_15 != width_30) or (width_5 != width_30):
            raise ValueError("One of the data sources has different crs")

        if (height_5 != height_15) or (height_15 != height_30) or (height_5 != height_30):
            raise ValueError("One of the data sources has different crs")

        # initialize the output raster
        output = np.full(data_5.shape, nodata, dtype=np.float32)

        # get where data is valid
        valid_5 = (data_5 != nodata_5) & ~np.isnan(data_5)
        valid_15 = (data_15 != nodata_15) & ~np.isnan(data_15)
        valid_30 = (data_30 != nodata_30) & ~np.isnan(data_30)
        valid = (valid_5 & valid_15 & valid_30)

        # Fill output
        output[valid] = (5*data_5[valid] + 10*data_15[valid] + 15*data_30[valid]) / 30


        return SoilGridsData(
            soil_data=output, 
            soil_meta= meta_5, 
            soil_crs = crs_5, 
            soil_transform=transform_5, 
            soil_width=width_5, 
            soil_height=height_5, 
            soil_nodata=nodata
        )


def _calculate_OM(soc, om_max: Optional[float] = OM_MAX):
    """Organic matter % from SOC in % (already converted by _load_soil_data_as_percentage): OM = 1.724 * SOC. Capped at om_max (None = no cap)."""
    om = 1.724 * np.asarray(soc, dtype=np.float32)
    return np.minimum(om, om_max) if om_max is not None else om


def _transform_soil_units(clay: np.ndarray, silt: np.ndarray, sand: np.ndarray, soc: np.ndarray):
    """# Transformation obtained from soilgrids FAQ page (https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html)

    Args:
        clay (np.ndarray): _description_
        silt (np.ndarray): _description_
        sand (np.ndarray): _description_
        soc (np.ndarray): _description_
    """
    
    clay_out = clay / 10  
    silt_out = silt / 10
    sand_out = sand / 10
    soc_out = soc / (10 * 10)

    return (clay_out, silt_out, sand_out, soc_out)

def _determine_usda_class_number_permeability(sand: np.ndarray, silt: np.ndarray, clay: np.ndarray):
    """Implementing USDA triangle from here: https://harvestmath.com/calculators/soil-type-calculator/.
    Classifies soil as one of the USDA texture classes and assigns the permeability class. The structure class s
    comes from the FAO texture class of Jones, Spoor & Thomasson (2003, Table 1), see STRUCTURE_BY_FAO_TEXTURE.

    Args:
        sand (np.ndarray): Percentage of sand in the soil
        silt (np.ndarray): Percentage of silt in the soil
        clay (np.ndarray): Percentage of clay in the soil

    Returns:
        tuple of 3 uint8 arrays (0 = nodata):
            texture: USDA texture code (keys of USDA_TEXTURE_CLASSES)
            structure_class: nomograph structure class s 1-4 (values of STRUCTURE_BY_FAO_TEXTURE)
            permeability_class: permeability class 1-6 (values of PERMEABILITY_CLASSES)
    """
    # Normalizing first the content of soils
    sand, silt, clay, valid = _normalize_sand_silt_clay(sand, silt, clay)

    # Categorizing every pixel according to USDA texture triangle
    # Order matters: np.select takes the first matching condition
    conditions_permeability = [
        (sand >= 85) & ((silt + 1.5*clay)<15),                                                          # Sand
        ((silt + 1.5*clay)>=15) & ((silt + 2*clay)<30),                                                 # Loamy sand
        ((clay >= 7) & (clay < 20) & (sand > 52) & ((silt + 2*clay) >= 30)) | ((clay < 7) & (silt < 50) & ((silt + 2*clay) >= 30)),     # Sandy loam
        (clay >=7) & (clay < 27) & (silt >=28) & (silt <50) & (sand <=52),                              # Loam   
        ((silt >=50) & (clay >=12) & (clay < 27)) | ((clay < 12) & (silt >=50) & (silt <80)),           # Silt loam
        (silt >= 80) & (clay < 12),                                                                     # Silt
        ((clay >=20) & (clay < 35) & (silt < 28) & (sand >45)),                                        # Sandy clay loam
        ((clay >=27) & (clay < 40) & (sand > 20) & (sand <=45)),                                        # Clay loam
        (clay >=27) & (clay < 40) & (sand <= 20),                                                       # Silty Clay loam
        (clay >=35) & (sand > 45),                                                                      # Sandy clay
        (clay >=40) & (silt >= 40),                                                                     # Silty clay
        (clay >=40) & (silt < 40) & (sand <=45),                                                        # Clay 
    ]

    texture = np.select(conditions_permeability, list(USDA_TEXTURE_CLASSES), default=0).astype(np.uint8)
    texture[~valid] = 0

    # Lookups by texture code (index 0 maps to 0, so nodata stays 0)
    permeability_class = PERMEABILITY_BY_TEXTURE[texture]

    # Structure class from the FAO texture class
    fao_texture = _determine_fao_texture_class(sand, clay, valid)
    structure_class = STRUCTURE_BY_FAO_TEXTURE_LUT[fao_texture]
    structure_class[texture == 0] = 0

    return (texture, structure_class, permeability_class)


def _determine_fao_texture_class(sand: np.ndarray, clay: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """FAO texture class of the European Soil Database (Jones, Spoor & Thomasson 2003, Table 1).

    1 coarse: clay < 18% and sand > 65%
    2 medium: clay < 35% and sand > 15%, or clay >= 18% if sand > 65%
    3 medium fine: clay < 35% and sand < 15%
    4 fine: clay 35-60%
    5 very fine: clay > 60%

    Args:
        sand (np.ndarray): Percentage of sand, normalized so sand + silt + clay = 100
        clay (np.ndarray): Percentage of clay, normalized so sand + silt + clay = 100
        valid (np.ndarray, optional): Pixels with data. Others get 0

    Returns:
        np.ndarray: uint8 FAO texture code (keys of FAO_TEXTURE_CLASSES), 0 = nodata
    """
    sand = np.asarray(sand, dtype=np.float32)
    clay = np.asarray(clay, dtype=np.float32)

    # Order matters: np.select takes the first matching condition
    conditions = [
        clay > 60,                      # Very fine
        clay >= 35,                     # Fine
        (clay < 35) & (sand < 15),      # Medium fine
        ((clay < 35) & (sand >= 15)) | ((clay > 18) & (sand >= 65) ),    # Medium
        (clay <= 18) & (sand >= 65)     # Coarse
    ]
    fao_texture = np.select(conditions, [5, 4, 3, 2, 1], default=0).astype(np.uint8)

    if valid is not None:
        fao_texture[~valid] = 0
    fao_texture[~(np.isfinite(sand) & np.isfinite(clay))] = 0
    return fao_texture


def _normalize_sand_silt_clay(sand: np.ndarray, silt: np.ndarray, clay: np.ndarray):
    """Normalizes content to a 100%

    Args:
        sand (np.ndarray): Percentage of sand in the soil
        silt (np.ndarray): Percentage of silt in the soil
        clay (np.ndarray): Percentage of clay in the soil
    """
    
    sand = np.asarray(sand, dtype=np.float32)
    silt = np.asarray(silt, dtype=np.float32)
    clay = np.asarray(clay, dtype=np.float32)
    total = sand + silt + clay
    valid = np.isfinite(total) & (total > 0)

    sand_scl, silt_scl, clay_scl = (np.zeros_like(sand) for _ in range(3))

    sand_scl[valid] = sand[valid] / total[valid] * 100
    silt_scl[valid] = silt[valid] / total[valid] * 100
    clay_scl[valid] = clay[valid] / total[valid] * 100

    return (sand_scl, silt_scl, clay_scl, valid)
