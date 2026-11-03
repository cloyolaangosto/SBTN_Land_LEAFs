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


##################################
######### MY K CODE ################
##################################
"""
So I've found 4 methods to estimate K. Currently, the best method, as defined by Yang et al. (2024) require data that I do not have, which is the permeability class and the soil structure class.
So for now, we're starting with K-EPIC, and then checking if we can estimate the other values, which seems to be the most cryptic in all publications so far.
"""

def calculate_K_Epic(clay_fp_list, silt_fp_list, sand_fp_list, soc_fp_list):
    # First all data is open
    with rasterio.open()

    K_EPIC = np.zeros_like([1000,1000])

    return K_EPIC