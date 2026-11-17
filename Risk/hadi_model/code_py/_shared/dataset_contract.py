"""Variable contract for the analysis-ready dataset.

Python port of ``code/01_data/dataset_contract.R``, plus the variable lists that
the two R model scripts define inline.

Naming convention, inherited from the R data pipeline:

* ``<var>``                    the raw grid-cell value (the causal forests use it)
* ``<var>__rescaled__log``     ``log( (x - min) / (max - min) * 100 + 1 )``
                               (the regressions use it, so the models are log-log)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import path_processed

LOG_SUFFIX = "__rescaled__log"


def logged(vars_: list[str]) -> list[str]:
    return [v + LOG_SUFFIX for v in vars_]


def analysis_ready_dataset_path() -> Path:
    return path_processed("final_dataset", "analysis_ready_data_regression_vars.rds")


# --------------------------------------------------------------------------- #
# Raw (linear-scale) variable names
# --------------------------------------------------------------------------- #
OUTCOME_RAW = "AllCrops2010YieldGapPercent"

TREATMENTS_RAW = [
    "soilErosionDebt",
    "soilCompactionDebt",
    "soilOrganicCarbonDebt",
    "soilWaterDebt",
    "treeCoverDebt",
]

CONTROLS_environment_timeDelta_climateTerra_RAW = [
    "pr_sum_mean__mean_delta",
    "tmmn_mean_mean__mean_delta",
    "tmmx_mean_mean__mean_delta",
    "srad_mean_mean__mean_delta",
]

CONTROLS_environment_climateTerra_RAW = [
    "pr_end_sum_mean__mean",
    "tmmn_end_mean_mean__mean",
    "tmmx_end_mean_mean__mean",
    "srad_end_mean_mean__mean",
]

CONTROLS_environment_terrain_RAW = [
    "gmted_elevation__mean",
    "gmted_slope__mean",
    "geomorpho90m_tri__mean",
]

# Categorical: never rescaled or logged
CONTROLS_environment_soilType = "soil_type__mode"

CONTROLS_environment_soilGrid_RAW = ["clay__mean", "sand__mean", "silt__mean"]

CONTROLS_environment_climateWc_RAW = [
    "wc_srad__mean",
    "wc_tavg__mean",
    "wc_prec__mean",
    "wc_meanTempWarmestQuarter__mean",
    "wc_maxTempWarmestMonth__mean",
    "wc_precWettestQuarter__mean",
    "aridity__mean",
]

CONTROLS_adaptation_RAW = [
    "nfer_crop_nh4_end_mean__mean",
    "nfer_crop_no3_end_mean__mean",
    "nmanure_app_crop_end_mean__mean",
    "ndep_noy_end_mean__mean",
    "ndep_nhx_end_mean__mean",
    "phosphorus_OlsenP_kgha1_World_Aug2022_ver__mean",
    "pesticide_APR_sum_2015_H__mean",
    "irrigation_AEI_ha_end__mean",
]

CONTROLS_socioeconomic_RAW = [
    "hdi_ca2010__mean",
    "gdp_ppp_ca2010__mean",
    "gdp_per_capita_ca2010__mean",
    "property_rights_protection",
    "enforcement_imputed",
    "electricityRural_ca2010",
    "phoneSub_ca2010",
    "agriShr_ca2010",
    "agriEmp_ca2010",
    "bayesian_corruption",
    "machinery_ca2010",
]

crop_type_dummy_vars = [
    "dominantCropGerberRay_barley",
    "dominantCropGerberRay_cassava",
    "dominantCropGerberRay_maize",
    "dominantCropGerberRay_oilpalm",
    "dominantCropGerberRay_rapeseed",
    "dominantCropGerberRay_rice",
    "dominantCropGerberRay_sorghum",
    "dominantCropGerberRay_soybean",
    "dominantCropGerberRay_sugarcane",
    "dominantCropGerberRay_wheat",
]

ROBUSTNESS_EXTRA_CATEGORICAL = "dominantFieldSizeCategory"

ROBUSTNESS_EXTRA_RAW = [
    "soilDepth__mean",
    "nitrogen__sum",
    "waterTableDepth",
    "roadDens__mean",
    "travel_time_to_cities",
    "agriShareOfGovtExpend_ca2010",
    "agriOrientIdxForGovtExpend_ca2010",
    "efw_ca2010",
]

# --------------------------------------------------------------------------- #
# Log-scale names used by the regressions (01_fit_regression_models)
# --------------------------------------------------------------------------- #
OUTCOME_LOG = OUTCOME_RAW + LOG_SUFFIX
TREATMENTS_LOG = logged(TREATMENTS_RAW)
CONTROLS_environment_timeDelta_climateTerra_LOG = logged(
    CONTROLS_environment_timeDelta_climateTerra_RAW
)
CONTROLS_environment_climateTerra_LOG = logged(CONTROLS_environment_climateTerra_RAW)
CONTROLS_environment_terrain_LOG = logged(CONTROLS_environment_terrain_RAW)
CONTROLS_environment_soilGrid_LOG = logged(CONTROLS_environment_soilGrid_RAW)
CONTROLS_environment_climateWc_LOG = logged(CONTROLS_environment_climateWc_RAW)
CONTROLS_adaptation_LOG = logged(CONTROLS_adaptation_RAW)
CONTROLS_socioeconomic_LOG = logged(CONTROLS_socioeconomic_RAW)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def read_rds(path: Path | str) -> pd.DataFrame:
    """Read an R ``.rds`` data frame.

    ``pyreadr`` converts R factors to pandas ``Categorical`` and reads R ``NA``
    as ``NaN``/``None``. Tibbles and ``data.frame`` objects both work. Objects that
    are not data frames (e.g. fitted ``fixest`` models) cannot be read.
    """
    import pyreadr

    result = pyreadr.read_r(str(path))
    return next(iter(result.values()))
