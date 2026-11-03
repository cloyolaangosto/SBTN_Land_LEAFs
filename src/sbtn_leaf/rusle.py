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
    # First, get all data opened and calculated weights
    clay_w_data = _calculate_weighted_soil_layers(clay_fp_list)
    silt_w_data = _calculate_weighted_soil_layers(silt_fp_list)
    sand_w_data = _calculate_weighted_soil_layers(sand_fp_list)
    soc_w_data = _calculate_weighted_soil_layers(soc_fp_list)

    # Transforming units into %
    clay = clay_w_data.soil_data * 10  # Transformation obtained from soilgrids FAQ page (https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html)
    silt = silt_w_data.soil_data * 10
    sand = sand_w_data.soil_data * 10
    soc = soc_w_data.soil_data * 10 * 10

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

    # Transforming units
    k_epic[valid] = k_epic[valid] * 0.1317

    # Applying Want et al 2024 correction
    k_epic_corrected  = (1.8722 * k_epic[valid]) - 0.05

    # Writing both files
    output_meta = clay_w_data.soil_meta.copy()
    output_meta.update(dtype="float32", count=1, nodata=clay_w_data.soil_nodata, compress="lzw")
    
    with rasterio.open(output_fp_k_epic, "w", **output_meta) as dst:
        dst.write(k_epic.astype("float32"), 1)

    with rasterio.open(output_fp_k_epic_corr, "w", **output_meta) as dst:
        dst.write(k_epic_corrected.astype("float32"), 1)

def _calculate_weighted_soil_layers(soil_data_fp_list, raster_band = 1) -> SoilGridsData:
    with (
        rasterio.open(soil_data_fp_list[0]) as src_data5,
        rasterio.open(soil_data_fp_list[1]) as src_data15,
        rasterio.open(soil_data_fp_list[2]) as src_data30,
    ):
        data_5 = src_data5.read(raster_band)
        meta_5 = src_data5.copy()         # Copy metadata for potential output.
        crs_5 = src_data5.crs             # Get the coordinate reference system of the first raster.
        transform_5 = src_data5.transform
        width_5 = src_data5.width
        height_5 = src_data5.height
        nodata_5 = src_data5.nodata

        data_15 = src_data15.read(raster_band)
        crs_15 = src_data15.crs             # Get the coordinate reference system of the first raster.
        transform_15 = src_data15.transform
        width_15 = src_data15.width
        height_15 = src_data15.height
        nodata_15 = src_data15.nodata

        data_30 = src_data30.read(raster_band)
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
        output = np.zeros_like(data_5)

        # get where data is valid
        valid_5 = (data_5 != nodata_5) & ~np.isnan(data_5)
        valid_15 = (data_15 != nodata_15) & ~np.isnan(data_15)
        valid_30 = (data_30 != nodata_30) & ~np.isnan(data_30)
        valid = (valid_5 & valid_15 & valid_30)

        # Fill output
        output = np.where(valid, (5*data_5 + 10*data_15 + 15*data_15)/30, nodata_5)

        return SoilGridsData(
            soil_data=output, 
            soil_meta= meta_5, 
            soil_crs = crs_5, 
            soil_transform=transform_5, 
            soil_width=width_5, 
            soil_height=height_5, 
            soil_nodata=nodata_5
        )