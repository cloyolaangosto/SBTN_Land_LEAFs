# Variable contract for the final analysis-ready manuscript dataset.
#


analysis_ready_dataset_path <- function() {
  path_processed("final_dataset", "analysis_ready_data_regression_vars.rds")
}

historical_final_dataset_path <- function() {
  path_processed("final_dataset", "analysis_ready_data_full.rds")
}

OUTCOME_LOG <- "AllCrops2010YieldGapPercent__rescaled__log"
OUTCOME_RAW <- "AllCrops2010YieldGapPercent"

TREATMENTS_RAW <- c(
  "soilErosionDebt",
  "soilCompactionDebt",
  "soilOrganicCarbonDebt",
  "soilWaterDebt",
  "treeCoverDebt"
)

TREATMENTS_LOG <- paste0(TREATMENTS_RAW, "__rescaled__log")

CONTROLS_environment_timeDelta_climateTerra_RAW <- c(
  "pr_sum_mean__mean_delta",
  "tmmn_mean_mean__mean_delta",
  "tmmx_mean_mean__mean_delta",
  "srad_mean_mean__mean_delta"
)

CONTROLS_environment_climateTerra_RAW <- c(
  "pr_end_sum_mean__mean",
  "tmmn_end_mean_mean__mean",
  "tmmx_end_mean_mean__mean",
  "srad_end_mean_mean__mean"
)

CONTROLS_environment_terrain_RAW <- c(
  "gmted_elevation__mean",
  "gmted_slope__mean",
  "geomorpho90m_tri__mean"
)

CONTROLS_environment_soilType <- "soil_type__mode"

CONTROLS_environment_soilGrid_RAW <- c(
  "clay__mean",
  "sand__mean",
  "silt__mean"
)

CONTROLS_environment_climateWc_RAW <- c(
  "wc_srad__mean",
  "wc_tavg__mean",
  "wc_prec__mean",
  "wc_meanTempWarmestQuarter__mean",
  "wc_maxTempWarmestMonth__mean",
  "wc_precWettestQuarter__mean",
  "aridity__mean"
)

CONTROLS_adaptation_RAW <- c(
  "nfer_crop_nh4_end_mean__mean",
  "nfer_crop_no3_end_mean__mean",
  "nmanure_app_crop_end_mean__mean",
  "ndep_noy_end_mean__mean",
  "ndep_nhx_end_mean__mean",
  "phosphorus_OlsenP_kgha1_World_Aug2022_ver__mean",
  "pesticide_APR_sum_2015_H__mean",
  "irrigation_AEI_ha_end__mean"
)

CONTROLS_socioeconomic_RAW <- c(
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
  "machinery_ca2010"
)

crop_type_dummy_vars <- c(
  "dominantCropGerberRay_barley",
  "dominantCropGerberRay_cassava",
  "dominantCropGerberRay_maize",
  "dominantCropGerberRay_oilpalm",
  "dominantCropGerberRay_rapeseed",
  "dominantCropGerberRay_rice",
  "dominantCropGerberRay_sorghum",
  "dominantCropGerberRay_soybean",
  "dominantCropGerberRay_sugarcane",
  "dominantCropGerberRay_wheat"
)

ROBUSTNESS_EXTRA_CATEGORICAL <- "dominantFieldSizeCategory"

ROBUSTNESS_EXTRA_RAW <- c(
  "soilDepth__mean",
  "nitrogen__sum",
  "waterTableDepth",
  "roadDens__mean",
  "travel_time_to_cities",
  "agriShareOfGovtExpend_ca2010",
  "agriOrientIdxForGovtExpend_ca2010",
  "efw_ca2010"
)

model_raw_vars <- c(
  OUTCOME_RAW,
  TREATMENTS_RAW,
  "actualYieldGerberRay",
  "attainableYieldGerberRay",
  CONTROLS_environment_timeDelta_climateTerra_RAW,
  CONTROLS_environment_climateTerra_RAW,
  CONTROLS_environment_terrain_RAW,
  CONTROLS_environment_soilGrid_RAW,
  CONTROLS_environment_climateWc_RAW,
  CONTROLS_adaptation_RAW,
  CONTROLS_socioeconomic_RAW,
  ROBUSTNESS_EXTRA_RAW
)

with_logged_pair <- function(vars) {
  c(vars, paste0(vars, "__rescaled__log"))
}

analysis_ready_keep_vars <- unique(c(
  "long",
  "lat",
  with_logged_pair(model_raw_vars),
  CONTROLS_environment_soilType,
  ROBUSTNESS_EXTRA_CATEGORICAL,
  crop_type_dummy_vars
))

analysis_ready_required_for_regression <- unique(c(
  OUTCOME_LOG,
  TREATMENTS_LOG,
  paste0(CONTROLS_environment_timeDelta_climateTerra_RAW, "__rescaled__log"),
  paste0(CONTROLS_environment_climateTerra_RAW, "__rescaled__log"),
  paste0(CONTROLS_environment_terrain_RAW, "__rescaled__log"),
  CONTROLS_environment_soilType,
  paste0(CONTROLS_environment_soilGrid_RAW, "__rescaled__log"),
  paste0(CONTROLS_environment_climateWc_RAW, "__rescaled__log"),
  paste0(CONTROLS_adaptation_RAW, "__rescaled__log"),
  paste0(CONTROLS_socioeconomic_RAW, "__rescaled__log"),
  "actualYieldGerberRay__rescaled__log",
  "attainableYieldGerberRay__rescaled__log",
  ROBUSTNESS_EXTRA_CATEGORICAL,
  "soilDepth__mean__rescaled__log",
  "nitrogen__sum__rescaled__log",
  "waterTableDepth__rescaled__log",
  "roadDens__mean__rescaled__log",
  "travel_time_to_cities__rescaled__log",
  "agriShareOfGovtExpend_ca2010__rescaled__log",
  "agriOrientIdxForGovtExpend_ca2010__rescaled__log",
  "efw_ca2010__rescaled__log",
  crop_type_dummy_vars,
  "long",
  "lat"
))

analysis_ready_required_for_causal_forest <- unique(c(
  "long",
  "lat",
  OUTCOME_RAW,
  TREATMENTS_RAW,
  "actualYieldGerberRay",
  "attainableYieldGerberRay",
  CONTROLS_environment_timeDelta_climateTerra_RAW,
  CONTROLS_environment_climateTerra_RAW,
  CONTROLS_environment_terrain_RAW,
  CONTROLS_environment_soilType,
  CONTROLS_environment_soilGrid_RAW,
  CONTROLS_environment_climateWc_RAW,
  CONTROLS_adaptation_RAW,
  CONTROLS_socioeconomic_RAW,
  crop_type_dummy_vars
))

validate_analysis_ready_columns <- function(df) {
  missing_keep <- setdiff(analysis_ready_keep_vars, names(df))
  missing_regression <- setdiff(analysis_ready_required_for_regression, names(df))
  missing_cf <- setdiff(analysis_ready_required_for_causal_forest, names(df))
  list(
    missing_keep = missing_keep,
    missing_regression = missing_regression,
    missing_causal_forest = missing_cf,
    ok = length(missing_keep) == 0 && length(missing_regression) == 0 && length(missing_cf) == 0
  )
}
