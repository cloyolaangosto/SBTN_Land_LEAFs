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
import sqlite3

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from tqdm.auto import tqdm

from affine import Affine
from rasterio.crs import CRS

from sbtn_leaf.paths import data_path, project_path
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

## HWSD v2.0 data, used for the soil structure class (s) and the profile permeability class (p)
# Harmonized World Soil Database v2.0 (FAO & IIASA 2023, https://doi.org/10.4060/cc3823en). Licence: CC BY-NC-SA 3.0 IGO
HWSD2_RASTER_FP = data_path("soil_weather", "HWSD_2", "HWSD2.tif")      # 30 arc-second grid of soil mapping unit (SMU) ids
HWSD2_DB_FP = data_path("soil_weather", "HWSD_2", "HWSD2.sqlite")       # soils of each SMU and their attributes
HWSD2_NODATA = 65535
RUSLE_OUTPUT_DIR = project_path("RUSLE", "outputs")
HWSD2_SMU_GRID_FP = RUSLE_OUTPUT_DIR / "hwsd2_smu_1km.tif"              # SMU ids on the SoilGrids 1 km grid

# FAO-90 major soil groups (first two letters of the HWSD v2 FAO90 code). Any other code is not a soil
# (water, glaciers, rock outcrops, dunes, salt flats, urban land, no data) and gets no s or p
FAO90_SOIL_GROUPS = (
    "AC", "AL", "AN", "AR", "AT", "CH", "CL", "CM", "FL", "FR", "GL", "GR", "GY", "HS",
    "KS", "LP", "LV", "LX", "NT", "PD", "PH", "PL", "PT", "PZ", "RG", "SC", "SN", "VR",
)

# Soil structure class s. Rule: topsoil structure of the European Soil Database (pedotransfer rule 422; King et al.
# 1994, section 3.4.4 and Table 20, the rule base published as Van Ranst et al. 1995):
#   1) permanent grass and semi-natural land: Good
#   2) the soil names of Table 20a: Good, whatever the land use
#   3) all other soils and uses: Normal
# Panagos et al. (2014, Table 1) turned these classes into the nomograph code s. Poor is never given to a topsoil
STRUCTURE_S_BY_ESDB_CLASS: Dict[str, int] = {
    "G": 1,     # Good -> very fine granular
    "N": 2,     # Normal -> fine granular
    "P": 3,     # Poor -> medium or coarse granular
    "H": 4,     # Humic or peaty topsoil -> blocky, platy or massive
}
# Soil names of Table 20a (FAO-Unesco 1974 legend, modified CEC 1985) as HWSD v2 FAO-90 soil units.
# "Bea" (Ando-Eutric Cambisol) has no FAO-90 unit, so it is left out. Two codes are misprinted in the scanned table
ESDB_GOOD_STRUCTURE_FAO90: Dict[str, str] = {
    "CMu": "Bh, Humic Cambisol",
    "LPk": "Eo, Orthic Rendzina",
    "PHc": "Hc, Calcaric Phaeozem",
    "PHh": "printed 'HL', read as Hh, Haplic Phaeozem",
    "PHl": "Hl, Luvic Phaeozem",
    "ANh": "To, Ochric Andosol",
    "ANz": "Tv, Vitric Andosol",
    "ATf": "printed 'pL', read as p, Plaggensol",
}
# Humic or peaty topsoil. The report does not print these rows of rule 422, so they are taken from its peat rule
# (Table 10: Histosols and two histic sub-units). Only the Histosols exist as FAO-90 units
ESDB_PEATY_FAO90_GROUPS = ("HS",)

# Profile permeability class p. HWSD v2 texture codes (database table D_TEXTURE_USDA) as PERMEABILITY_CLASSES names
HWSD2_TEXTURE_NAMES: Dict[int, str] = {
    1: "clay",              # Clay (heavy)
    2: "silty clay",
    3: "clay",              # Clay (light)
    4: "silty clay loam",
    5: "clay loam",
    6: "silt",
    7: "silt loam",
    8: "sandy clay",
    9: "loam",
    10: "sandy clay loam",
    11: "sandy loam",
    12: "loamy sand",
    13: "sand",
}
# Layers that restrict water movement (RUSLE2 science documentation, section 4.1.1.4: "restricting layers such as
# rock, claypan, or fragipan"). They give the slowest class
PERMEABILITY_VERY_SLOW = 6
HWSD2_IMPERMEABLE_LAYER_CLASSES = (3, 4)    # ESDB attribute, Europe only: 3 = between 40 and 80 cm, 4 = within 40 cm
HWSD2_PAN_PHASES: Dict[int, str] = {        # soil phases defined as a cemented or slowly permeable pan within 100 cm
    4: "Petrocalcic",
    5: "Petrogypsic",
    6: "Petroferric",
    8: "Fragipan",
    9: "Duripan",
}
# Optional rule: reference drainage classes with a shallow water table (poorly and very poorly drained)
HWSD2_WET_DRAINAGE_CLASSES = ("P", "VP")

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


##################################
######### MY K CODE ################
##################################
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

    # Texture number (used as structure proxy) and permeability class.
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
    Classifies soil as one of the USDA texture classes, and assigns number and permeability class

    Args:
        sand (np.ndarray): Percentage of sand in the soil
        silt (np.ndarray): Percentage of silt in the soil
        clay (np.ndarray): Percentage of clay in the soil

    Returns:
        tuple of 3 uint8 arrays (0 = nodata):
            texture: USDA texture code (keys of USDA_TEXTURE_CLASSES)
            texture_number: texture number (keys of NUMBER_TEXTURE_CLASSES)
            permeability_class: permeability class 1-6 (values of PERMEABILITY_CLASSES)
    """
    # Normalizing first the content of soils
    sand, silt, clay, valid = _normalize_sand_silt_clay(sand, silt, clay)

    # Categorizing every pixel according to USDA texture triangle
    # Order matters: np.select takes the first matching condition
    conditions = [
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

    texture = np.select(conditions, list(USDA_TEXTURE_CLASSES), default=0).astype(np.uint8)
    texture[~valid] = 0

    # Lookups by texture code (index 0 maps to 0, so nodata stays 0)
    texture_number = NUMBER_BY_TEXTURE[texture]
    permeability_class = PERMEABILITY_BY_TEXTURE[texture]

    return (texture, texture_number, permeability_class)


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


##################################
#### HWSD v2: s AND p CLASSES ####
##################################
"""
SoilGrids has no soil structure class (s) and no profile permeability class (p), which the K nomograph needs.
Both are taken here from the Harmonized World Soil Database v2.0 (HWSD v2). Its raster gives a soil mapping unit
(SMU) per 30 arc-second cell, and its database lists the soils of each SMU with their share, FAO-90 name, phases
and the texture of 7 depth layers.

- s follows the topsoil structure rule of the European Soil Database (soil name + land use), which Panagos et al.
  (2014) used for the K map of Europe. See the notes above STRUCTURE_S_BY_ESDB_CLASS.
- p is an add-on. It is the slowest texture-based permeability class within the soil profile, raised to very slow
  where HWSD v2 reports an impermeable layer or a pan.

Each SMU gets the class that covers the largest share of it, and each 1 km cell gets the SMU found at its centre.
"""

def create_HWSD2_SMU_grid(output_fp: PathLike = HWSD2_SMU_GRID_FP, ref_fp: PathLike = CLAY_LIST[0], hwsd_raster_fp: PathLike = HWSD2_RASTER_FP) -> Path:
    """Puts the HWSD v2 soil mapping unit (SMU) ids on the grid of the SoilGrids 1 km layers.

    HWSD v2 is a 30 arc-second lat/lon grid and the SoilGrids layers are in Homolosine at 1 km. Each 1 km cell gets
    the SMU found at its centre (nearest resampling), so any cell can be traced back to one SMU of the database.
    Mode resampling (the SMU covering most of the cell) takes about 100 times longer and changes the s class of
    only 0.4-2% of the cells, all on mapping unit boundaries.

    Args:
        output_fp (PathLike): GeoTIFF to write (uint16, nodata 65535)
        ref_fp (PathLike): raster that defines the output grid
        hwsd_raster_fp (PathLike): HWSD v2 raster of SMU ids
    """
    print("Creating HWSD v2 soil mapping unit grid...")
    output_fp = Path(output_fp)
    output_fp.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ref_fp) as ref, rasterio.open(hwsd_raster_fp) as src:
        output_meta = ref.meta.copy()
        output_meta.update(driver="GTiff", dtype="uint16", count=1, nodata=HWSD2_NODATA, compress="lzw", tiled=True, blockxsize=512, blockysize=512)

        with rasterio.open(output_fp, "w", **output_meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_nodata=src.nodata,
                dst_nodata=HWSD2_NODATA,
                resampling=Resampling.nearest,
                num_threads=4,
            )

    print(f"     HWSD v2 SMU grid saved into {output_fp}")
    return output_fp


def calculate_s_HWSD2(output_fp: PathLike, land_use: str = "cultivated", smu_grid_fp: PathLike = HWSD2_SMU_GRID_FP, db_fp: PathLike = HWSD2_DB_FP) -> Path:
    """Soil structure class s of the K nomograph from the HWSD v2 soil names, on the SoilGrids 1 km grid.

    Classes: 1 very fine granular, 2 fine granular, 3 medium or coarse granular, 4 blocky, platy or massive, 0 nodata.
    The rule is the topsoil structure rule of the European Soil Database (see STRUCTURE_S_BY_ESDB_CLASS):
        - Histosols: humic or peaty topsoil (4)
        - land_use "grass_seminatural": all other soils Good (1)
        - land_use "cultivated": the soil units of ESDB_GOOD_STRUCTURE_FAO90 Good (1), all other soils Normal (2)

    K is defined for a tilled fallow plot, so "cultivated" is the consistent choice for K.

    Args:
        output_fp (PathLike): GeoTIFF to write (uint8)
        land_use (str): "cultivated" or "grass_seminatural"
        smu_grid_fp (PathLike): HWSD v2 SMU ids on the 1 km grid. Created with create_HWSD2_SMU_grid if missing
        db_fp (PathLike): HWSD v2 SQLite database
    """
    print(f"Calculating soil structure class (s) from HWSD v2 for {land_use} land")
    if not Path(smu_grid_fp).exists():
        create_HWSD2_SMU_grid(smu_grid_fp)

    soils = _load_HWSD2_soils(db_fp)
    s_by_smu = _HWSD2_dominant_class(soils, _HWSD2_structure_class(soils, land_use))

    shares = _write_HWSD2_class_raster(s_by_smu, output_fp, smu_grid_fp)
    print("     Share of the classified area per s class: " + ", ".join(f"{c}: {v:.1%}" for c, v in shares.items()))
    print(f"     s saved into {output_fp}")
    return Path(output_fp)


def calculate_p_HWSD2(output_fp: PathLike, profile_depth_cm: float = 100, use_restricting_layers: bool = True, use_drainage: bool = False, smu_grid_fp: PathLike = HWSD2_SMU_GRID_FP, db_fp: PathLike = HWSD2_DB_FP) -> Path:
    """Profile permeability class p of the K nomograph from HWSD v2, on the SoilGrids 1 km grid.

    Classes: 1 rapid, 2 moderate to rapid, 3 moderate, 4 slow to moderate, 5 slow, 6 very slow, 0 nodata.
    The nomograph asks for the permeability of the whole profile, not of the topsoil, so p is built in 3 steps:
        1) Slowest class among the HWSD v2 layers above profile_depth_cm, each from its texture class
           (PERMEABILITY_CLASSES). This picks up clay-rich subsoils under a coarser topsoil.
        2) use_restricting_layers: very slow (6) where HWSD v2 reports an impermeable layer within 80 cm (ESDB
           attribute, so Europe only) or a fragipan, duripan or petrocalcic, petrogypsic or petroferric horizon.
        3) use_drainage (off by default): very slow (6) for poorly and very poorly drained soils. The HWSD v2
           drainage class is a reference value for flat land, inferred from soil type and topsoil texture, and
           there is no published drainage-to-p table. This step is therefore our own assumption.

    Shallow rock (Leptosols, lithic phase) does not change p.

    Args:
        output_fp (PathLike): GeoTIFF to write (uint8)
        profile_depth_cm (float): depth of the profile. HWSD v2 layers start at 0, 20, 40, 60, 80, 100 and 150 cm
        use_restricting_layers (bool): apply step 2
        use_drainage (bool): apply step 3
        smu_grid_fp (PathLike): HWSD v2 SMU ids on the 1 km grid. Created with create_HWSD2_SMU_grid if missing
        db_fp (PathLike): HWSD v2 SQLite database
    """
    print("Calculating profile permeability class (p) from HWSD v2")
    if not Path(smu_grid_fp).exists():
        create_HWSD2_SMU_grid(smu_grid_fp)

    soils = _load_HWSD2_soils(db_fp, profile_depth_cm)
    p_by_smu = _HWSD2_dominant_class(soils, _HWSD2_permeability_class(soils, use_restricting_layers, use_drainage))

    shares = _write_HWSD2_class_raster(p_by_smu, output_fp, smu_grid_fp)
    print("     Share of the classified area per p class: " + ", ".join(f"{c}: {v:.1%}" for c, v in shares.items()))
    print(f"     p saved into {output_fp}")
    return Path(output_fp)


def _load_HWSD2_soils(db_fp: PathLike = HWSD2_DB_FP, profile_depth_cm: float = 100) -> pd.DataFrame:
    """Loads one row per soil of each HWSD v2 mapping unit.

    Returns:
        pd.DataFrame: smu (mapping unit id), seq (order of the soil in the unit, 1 = dominant), share (% of the unit),
            fao90 (FAO-90 soil unit code), phase1, phase2, il (impermeable layer class), drainage, and p_texture
            (slowest texture-based permeability class of the layers above profile_depth_cm, 0 if none has a texture)
    """
    query = """
        SELECT HWSD2_SMU_ID AS smu, SEQUENCE AS seq, SHARE AS share, FAO90 AS fao90, PHASE1 AS phase1,
               PHASE2 AS phase2, IL AS il, DRAINAGE AS drainage, TOPDEP AS topdep, TEXTURE_USDA AS texture
        FROM HWSD2_LAYERS
    """
    con = sqlite3.connect(Path(db_fp).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        layers = pd.read_sql_query(query, con)
    finally:
        con.close()

    # Permeability class of each layer from its texture class. Layers without texture (no soil at that depth) get 0
    p_by_texture = {code: PERMEABILITY_CLASSES[name] for code, name in HWSD2_TEXTURE_NAMES.items()}
    layers["p_texture"] = layers["texture"].map(p_by_texture).fillna(0)

    # The soil attributes are repeated on every layer row, so the first one is kept
    soils = (
        layers[layers["topdep"] < profile_depth_cm]
        .groupby(["smu", "seq"], as_index=False)
        .agg(
            share=("share", "first"),
            fao90=("fao90", "first"),
            phase1=("phase1", "first"),
            phase2=("phase2", "first"),
            il=("il", "first"),
            drainage=("drainage", "first"),
            p_texture=("p_texture", "max"),
        )
    )
    return soils


def _HWSD2_is_soil(soils: pd.DataFrame) -> np.ndarray:
    """True for the rows that are a soil, False for water, glaciers, rock outcrops, dunes, urban land and no data."""
    return soils["fao90"].fillna("").str[:2].isin(FAO90_SOIL_GROUPS).to_numpy()


def _HWSD2_structure_class(soils: pd.DataFrame, land_use: str = "cultivated") -> np.ndarray:
    """Soil structure class s (1-4) of each HWSD v2 soil, 0 where it is not a soil. Rule in calculate_s_HWSD2."""
    if land_use not in ("cultivated", "grass_seminatural"):
        raise ValueError(f"land_use must be 'cultivated' or 'grass_seminatural', got '{land_use}'")

    fao90 = soils["fao90"].fillna("")
    is_soil = _HWSD2_is_soil(soils)
    peaty = fao90.str[:2].isin(ESDB_PEATY_FAO90_GROUPS).to_numpy()
    if land_use == "cultivated":
        good = fao90.isin(list(ESDB_GOOD_STRUCTURE_FAO90)).to_numpy()
    else:
        good = is_soil

    s = np.zeros(len(soils), dtype=np.uint8)
    s[is_soil] = STRUCTURE_S_BY_ESDB_CLASS["N"]
    s[is_soil & good] = STRUCTURE_S_BY_ESDB_CLASS["G"]
    s[is_soil & peaty] = STRUCTURE_S_BY_ESDB_CLASS["H"]
    return s


def _HWSD2_permeability_class(soils: pd.DataFrame, use_restricting_layers: bool = True, use_drainage: bool = False) -> np.ndarray:
    """Profile permeability class p (1-6) of each HWSD v2 soil, 0 where it is not a soil. Rule in calculate_p_HWSD2."""
    p = soils["p_texture"].to_numpy(dtype=np.uint8, copy=True)
    p[~_HWSD2_is_soil(soils)] = 0
    has_class = p > 0

    if use_restricting_layers:
        pan_phases = list(HWSD2_PAN_PHASES)
        restricted = (
            soils["il"].isin(HWSD2_IMPERMEABLE_LAYER_CLASSES)
            | soils["phase1"].isin(pan_phases)
            | soils["phase2"].isin(pan_phases)
        ).to_numpy()
        p[has_class & restricted] = PERMEABILITY_VERY_SLOW

    if use_drainage:
        wet = soils["drainage"].isin(HWSD2_WET_DRAINAGE_CLASSES).to_numpy()
        p[has_class & wet] = PERMEABILITY_VERY_SLOW

    return p


def _HWSD2_dominant_class(soils: pd.DataFrame, soil_class: np.ndarray) -> np.ndarray:
    """Class that covers the largest share of each mapping unit.

    Rows with class 0 (not a soil) are left out, so a unit gets the class of its soils even when water or rock is
    the largest part of it. A tie goes to the class of the soil listed first in the unit.

    Returns:
        np.ndarray: uint8 lookup array indexed by SMU id, 0 where the unit has no soil
    """
    table = pd.DataFrame({
        "smu": soils["smu"].to_numpy(),
        "seq": soils["seq"].to_numpy(),
        "share": soils["share"].to_numpy(),
        "soil_class": soil_class,
    })
    table = table[table["soil_class"] > 0]

    by_class = table.groupby(["smu", "soil_class"], as_index=False).agg(share=("share", "sum"), seq=("seq", "min"))
    by_class = by_class.sort_values(["smu", "share", "seq"], ascending=[True, False, True])
    dominant = by_class.drop_duplicates("smu")

    lookup = np.zeros(HWSD2_NODATA + 1, dtype=np.uint8)
    lookup[dominant["smu"].to_numpy()] = dominant["soil_class"].to_numpy()
    return lookup


def _write_HWSD2_class_raster(class_by_smu: np.ndarray, output_fp: PathLike, smu_grid_fp: PathLike = HWSD2_SMU_GRID_FP) -> Dict[int, float]:
    """Writes the class of the SMU of every cell of the SMU grid (uint8, 0 = nodata).

    Returns:
        Dict[int, float]: share of the classified cells per class. The grid is equal-area, so it is also the area share
    """
    output_fp = Path(output_fp)
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    counts = np.zeros(256, dtype=np.int64)

    with rasterio.open(smu_grid_fp) as src:
        output_meta = src.meta.copy()
        output_meta.update(dtype="uint8", count=1, nodata=0, compress="lzw", tiled=True, blockxsize=512, blockysize=512)

        with rasterio.open(output_fp, "w", **output_meta) as dst:
            for row in range(0, src.height, 2048):
                window = Window(0, row, src.width, min(2048, src.height - row))
                classes = class_by_smu[src.read(1, window=window)]
                dst.write(classes, 1, window=window)
                counts += np.bincount(classes.ravel(), minlength=256)

    classified = counts[1:].sum()
    if classified == 0:
        return {}
    return {int(c): float(counts[c] / classified) for c in np.nonzero(counts)[0] if c > 0}
