# Fit the regression models used for the manuscript coefficient plots.
#
# This script estimates the baseline yield-gap specification and the reported
# robustness checks:
#   - baseline yield-gap model and Fig. 2 model-specification objects
#   - nonlinear LD/adaptation average-slope objects used in Fig. 2
#   - actual-yield and attainable-yield models used in Fig. 3 / Supp. Fig. 6
#   - spatial-resolution robustness inputs used in Supp. Fig. 5
#


library(dplyr)
library(fixest)
library(readr)
library(tibble)

source(file.path("code", "_shared", "analysis_paths.R"))
source(file.path("code", "01_data", "dataset_contract.R"))

dataset_path <- analysis_ready_dataset_path()
model_dir <- path_processed("models", "main_10km")
model_dir_hc1 <- path_processed("models", "hc1")
model_dir_conley100 <- path_processed("models", "conley_100km")
coarser_dir <- path_intermediate("coarser_resolution", "loglog")
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir_hc1, recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir_conley100, recursive = TRUE, showWarnings = FALSE)

OUTCOME <- "AllCrops2010YieldGapPercent__rescaled__log"

TREATMENTS <- c(
  "soilErosionDebt__rescaled__log",
  "soilCompactionDebt__rescaled__log",
  "soilOrganicCarbonDebt__rescaled__log",
  "soilWaterDebt__rescaled__log",
  "treeCoverDebt__rescaled__log"
)

CONTROLS_environment_timeDelta_climateTerra <- c(
  "pr_sum_mean__mean_delta__rescaled__log",
  "tmmn_mean_mean__mean_delta__rescaled__log",
  "tmmx_mean_mean__mean_delta__rescaled__log",
  "srad_mean_mean__mean_delta__rescaled__log"
)

CONTROLS_environment_climateTerra <- c(
  "pr_end_sum_mean__mean__rescaled__log",
  "tmmn_end_mean_mean__mean__rescaled__log",
  "tmmx_end_mean_mean__mean__rescaled__log",
  "srad_end_mean_mean__mean__rescaled__log"
)

CONTROLS_environment_terrain <- c(
  "gmted_elevation__mean__rescaled__log",
  "gmted_slope__mean__rescaled__log",
  "geomorpho90m_tri__mean__rescaled__log"
)

CONTROLS_environment_soilType <- "soil_type__mode"

CONTROLS_environment_soilGrid <- c(
  "clay__mean__rescaled__log",
  "sand__mean__rescaled__log",
  "silt__mean__rescaled__log"
)

CONTROLS_environment_climateWc <- c(
  "wc_srad__mean__rescaled__log",
  "wc_tavg__mean__rescaled__log",
  "wc_prec__mean__rescaled__log",
  "wc_meanTempWarmestQuarter__mean__rescaled__log",
  "wc_maxTempWarmestMonth__mean__rescaled__log",
  "wc_precWettestQuarter__mean__rescaled__log",
  "aridity__mean__rescaled__log"
)

CONTROLS_adaptation <- c(
  "nfer_crop_nh4_end_mean__mean__rescaled__log",
  "nfer_crop_no3_end_mean__mean__rescaled__log",
  "nmanure_app_crop_end_mean__mean__rescaled__log",
  "ndep_noy_end_mean__mean__rescaled__log",
  "ndep_nhx_end_mean__mean__rescaled__log",
  "phosphorus_OlsenP_kgha1_World_Aug2022_ver__mean__rescaled__log",
  "pesticide_APR_sum_2015_H__mean__rescaled__log",
  "irrigation_AEI_ha_end__mean__rescaled__log"
)

CONTROLS_socioeconomic <- c(
  "hdi_ca2010__mean__rescaled__log",
  "gdp_ppp_ca2010__mean__rescaled__log",
  "gdp_per_capita_ca2010__mean__rescaled__log",
  "property_rights_protection__rescaled__log",
  "enforcement_imputed__rescaled__log",
  "electricityRural_ca2010__rescaled__log",
  "phoneSub_ca2010__rescaled__log",
  "agriShr_ca2010__rescaled__log",
  "agriEmp_ca2010__rescaled__log",
  "bayesian_corruption__rescaled__log",
  "machinery_ca2010__rescaled__log"
)

natural_controls <- c(
  CONTROLS_environment_climateWc,
  CONTROLS_environment_climateTerra,
  CONTROLS_environment_timeDelta_climateTerra,
  CONTROLS_environment_terrain,
  CONTROLS_environment_soilType,
  CONTROLS_environment_soilGrid
)

f <- as.formula(paste(
  OUTCOME, "~",
  paste(TREATMENTS, collapse = " + "), "+",
  paste(natural_controls, collapse = " + "), "+",
  paste(CONTROLS_adaptation, collapse = " + "), "+",
  paste(CONTROLS_socioeconomic, collapse = " + ")
))

make_formula <- function(outcome = OUTCOME, controls = character()) {
  rhs <- c(TREATMENTS, controls)
  as.formula(paste(outcome, "~", paste(rhs, collapse = " + ")))
}

fit_and_save <- function(name, formula, df_ok, vcov = NULL, out_dir = model_dir) {
  model <- fixest::feols(formula, data = df_ok, vcov = vcov)
  readr::write_rds(model, file.path(out_dir, name))
  model
}

winsorize <- function(x, lower_quantile = 0.025, upper_quantile = 0.975) {
  if (!is.numeric(x)) {
    return(x)
  }
  q <- quantile(x, probs = c(lower_quantile, upper_quantile), na.rm = TRUE)
  pmin(pmax(x, q[[1]]), q[[2]])
}


assert_files_exist(dataset_path)
df_ok <- readr::read_rds(dataset_path)

crop_cols <- names(df_ok)[grepl("^dominantCropGerberRay_", names(df_ok))]
df_ok <- df_ok |>
  mutate(
    across(all_of(crop_cols), as.factor)
  )

readr::write_rds(
  list(
    OUTCOME = OUTCOME,
    TREATMENTS = TREATMENTS,
    CONTROLS_environment_timeDelta_climateTerra = CONTROLS_environment_timeDelta_climateTerra,
    CONTROLS_environment_climateTerra = CONTROLS_environment_climateTerra,
    CONTROLS_environment_terrain = CONTROLS_environment_terrain,
    CONTROLS_environment_soilType = CONTROLS_environment_soilType,
    CONTROLS_environment_soilGrid = CONTROLS_environment_soilGrid,
    CONTROLS_environment_climateWc = CONTROLS_environment_climateWc,
    CONTROLS_adaptation = CONTROLS_adaptation,
    CONTROLS_socioeconomic = CONTROLS_socioeconomic
  ),
  file.path(model_dir, "model_spec_variables.rds")
)
readr::write_rds(
  list(
    OUTCOME = OUTCOME,
    TREATMENTS = TREATMENTS,
    CONTROLS_environment_timeDelta_climateTerra = CONTROLS_environment_timeDelta_climateTerra,
    CONTROLS_environment_climateTerra = CONTROLS_environment_climateTerra,
    CONTROLS_environment_terrain = CONTROLS_environment_terrain,
    CONTROLS_environment_soilType = CONTROLS_environment_soilType,
    CONTROLS_environment_soilGrid = CONTROLS_environment_soilGrid,
    CONTROLS_environment_climateWc = CONTROLS_environment_climateWc,
    CONTROLS_adaptation = CONTROLS_adaptation,
    CONTROLS_socioeconomic = CONTROLS_socioeconomic
  ),
  file.path(model_dir_conley100, "model_spec_variables.rds")
)

vcov_main <- NULL

mf <- fit_and_save("model_baseline.rds", f, df_ok, vcov_main)

fit_and_save("model_baseline.rds", f, df_ok, vcov = "HC1", out_dir = model_dir_hc1)
fit_and_save("model_baseline.rds", f, df_ok, vcov = conley(cutoff = 100), out_dir = model_dir_conley100)

# Winsorized robustness model: cap numeric variables and re-estimate the
# baseline specification.
df_ok_winsor <- df_ok |>
  mutate(across(where(is.numeric), winsorize))
mf_winsor <- fit_and_save("model_winsorized.rds", f, df_ok_winsor, vcov_main)
fit_and_save("model_winsorized.rds", f, df_ok_winsor, vcov = conley(cutoff = 100), out_dir = model_dir_conley100)

mf_noControls <- fit_and_save("model_no_controls.rds", make_formula(controls = character()), df_ok, vcov_main)
mf_naturalControls <- fit_and_save("model_natural_controls.rds", make_formula(controls = natural_controls), df_ok, vcov_main)
mf_naturalControls_adaptControls <- fit_and_save(
  "model_natural_adaptation_controls.rds",
  make_formula(controls = c(natural_controls, CONTROLS_adaptation)),
  df_ok,
  vcov_main
)
mf_naturalControls_adaptControls_socioeconControls <- fit_and_save(
  "model_full_controls.rds",
  make_formula(controls = c(natural_controls, CONTROLS_adaptation, CONTROLS_socioeconomic)),
  df_ok,
  vcov_main
)

fit_and_save("model_no_controls.rds", make_formula(controls = character()), df_ok, vcov = conley(cutoff = 100), out_dir = model_dir_conley100)
fit_and_save("model_natural_controls.rds", make_formula(controls = natural_controls), df_ok, vcov = conley(cutoff = 100), out_dir = model_dir_conley100)
fit_and_save(
  "model_natural_adaptation_controls.rds",
  make_formula(controls = c(natural_controls, CONTROLS_adaptation)),
  df_ok,
  vcov = conley(cutoff = 100),
  out_dir = model_dir_conley100
)
fit_and_save(
  "model_full_controls.rds",
  make_formula(controls = c(natural_controls, CONTROLS_adaptation, CONTROLS_socioeconomic)),
  df_ok,
  vcov = conley(cutoff = 100),
  out_dir = model_dir_conley100
)

f_addLongLatPoly <- update(f, . ~ . + long + lat + I(long^2) + I(lat^2) + I(long * lat))
mf_addLongLatPoly <- fit_and_save("model_long_lat_polynomial.rds", f_addLongLatPoly, df_ok, vcov_main)
fit_and_save("model_long_lat_polynomial.rds", f_addLongLatPoly, df_ok, vcov = conley(cutoff = 100), out_dir = model_dir_conley100)

f_rmSoilGrid <- update(f, as.formula(paste(". ~ . -", paste(CONTROLS_environment_soilGrid, collapse = " - "))))
mf_rmSoilGrid <- fit_and_save("model_without_soilgrid_controls.rds", f_rmSoilGrid, df_ok, vcov_main)
fit_and_save("model_without_soilgrid_controls.rds", f_rmSoilGrid, df_ok, vcov = conley(cutoff = 100), out_dir = model_dir_conley100)

f_addCropType_RayGerber <- update(
  f,
  . ~ . +
    dominantCropGerberRay_barley + dominantCropGerberRay_cassava +
    dominantCropGerberRay_maize + dominantCropGerberRay_oilpalm +
    dominantCropGerberRay_rapeseed + dominantCropGerberRay_rice +
    dominantCropGerberRay_sorghum + dominantCropGerberRay_soybean +
    dominantCropGerberRay_sugarcane + dominantCropGerberRay_wheat
)
mf_addCropType_RayGerber <- fit_and_save("model_yield_gap_crop_type_controls.rds", f_addCropType_RayGerber, df_ok, vcov_main)
readr::write_rds(mf_addCropType_RayGerber, file.path(model_dir, "model_crop_type_controls.rds"))
mf_conley_addCropType_RayGerber <- fit_and_save("model_yield_gap_crop_type_controls.rds", f_addCropType_RayGerber, df_ok, vcov = conley(cutoff = 100), out_dir = model_dir_conley100)
readr::write_rds(mf_conley_addCropType_RayGerber, file.path(model_dir_conley100, "model_crop_type_controls.rds"))

# Additional covariate and exclusion robustness specifications used in
# Supplementary Fig. 3.
f_addFieldSize <- update(f, . ~ . + dominantFieldSizeCategory)
fit_and_save("model_field_size_control.rds", f_addFieldSize, df_ok, vcov_main)

f_addSoilDepth <- update(f, . ~ . + soilDepth__mean__rescaled__log)
fit_and_save("model_soil_depth_control.rds", f_addSoilDepth, df_ok, vcov_main)

f_addSoilN <- update(f, . ~ . + nitrogen__sum__rescaled__log)
fit_and_save("model_soil_n_control.rds", f_addSoilN, df_ok, vcov_main)

f_addWTD <- update(f, . ~ . + waterTableDepth__rescaled__log)
fit_and_save("model_water_table_depth_control.rds", f_addWTD, df_ok, vcov_main)

f_addRoadDensity <- update(f, . ~ . + roadDens__mean__rescaled__log)
fit_and_save("model_road_density_control.rds", f_addRoadDensity, df_ok, vcov_main)

f_addTravelCities <- update(f, . ~ . + travel_time_to_cities__rescaled__log)
fit_and_save("model_travel_time_to_cities_control.rds", f_addTravelCities, df_ok, vcov_main)

f_rmMachine <- update(f, . ~ . - machinery_ca2010__rescaled__log)
fit_and_save("model_without_machinery_control.rds", f_rmMachine, df_ok, vcov_main)

f_addAgriExpend <- update(f, . ~ . + agriShareOfGovtExpend_ca2010__rescaled__log + agriOrientIdxForGovtExpend_ca2010__rescaled__log)
fit_and_save("model_agricultural_expenditure_control.rds", f_addAgriExpend, df_ok, vcov_main)

f_addEconFree <- update(f, . ~ . + efw_ca2010__rescaled__log)
fit_and_save("model_economic_freedom_control.rds", f_addEconFree, df_ok, vcov_main)

f_actualYieldGerberRay <- update(f, actualYieldGerberRay__rescaled__log ~ .)
mf_actualYieldGerberRay <- fit_and_save("model_actual_yield.rds", f_actualYieldGerberRay, df_ok, vcov_main)

f_attainYieldGerberRay <- update(f, attainableYieldGerberRay__rescaled__log ~ .)
mf_attainYieldGerberRay <- fit_and_save("model_attainable_yield.rds", f_attainYieldGerberRay, df_ok, vcov_main)

f_actualYieldGerberRay_addCropType_RayGerber <- update(f_addCropType_RayGerber, actualYieldGerberRay__rescaled__log ~ .)
fit_and_save("model_actual_yield_crop_type_controls.rds", f_actualYieldGerberRay_addCropType_RayGerber, df_ok, vcov_main)

f_attainYieldGerberRay_addCropType_RayGerber <- update(f_addCropType_RayGerber, attainableYieldGerberRay__rescaled__log ~ .)
fit_and_save("model_attainable_yield_crop_type_controls.rds", f_attainYieldGerberRay_addCropType_RayGerber, df_ok, vcov_main)

if (allow_heavy("LDYG_ALLOW_AVG_SLOPES")) {
  library(marginaleffects)

  TREATMENTS_sq <- paste0("I(", TREATMENTS, "^2)")
  f_addQuadraticLDs <- update(f, as.formula(paste(". ~ . +", paste(TREATMENTS_sq, collapse = " + "))))
  mf_addQuadraticLDs <- fit_and_save("model_quadratic_land_degradation.rds", f_addQuadraticLDs, df_ok, vcov_main)
  readr::write_rds(
    marginaleffects::avg_slopes(mf_addQuadraticLDs, variables = TREATMENTS),
    file.path(model_dir, "model_quadratic_land_degradation_average_slope.rds")
  )
  mf_conley_addQuadraticLDs <- fit_and_save("model_quadratic_land_degradation.rds", f_addQuadraticLDs, df_ok, vcov = conley(cutoff = 100), out_dir = model_dir_conley100)
  readr::write_rds(
    marginaleffects::avg_slopes(mf_conley_addQuadraticLDs, variables = TREATMENTS),
    file.path(model_dir_conley100, "model_quadratic_land_degradation_average_slope.rds")
  )

  interactions_LDs <- unlist(lapply(2:length(TREATMENTS), function(i) combn(TREATMENTS, i, paste, collapse = ":")))
  f_addInteractionLDs <- update(f, as.formula(paste(". ~ . +", paste(interactions_LDs, collapse = " + "))))
  mf_addInteractionLDs <- fit_and_save("model_interaction_land_degradation.rds", f_addInteractionLDs, df_ok, vcov_main)
  readr::write_rds(
    marginaleffects::avg_slopes(mf_addInteractionLDs, variables = TREATMENTS),
    file.path(model_dir, "model_interaction_land_degradation_average_slope.rds")
  )
  mf_conley_addInteractionLDs <- fit_and_save("model_interaction_land_degradation.rds", f_addInteractionLDs, df_ok, vcov = conley(cutoff = 100), out_dir = model_dir_conley100)
  readr::write_rds(
    marginaleffects::avg_slopes(mf_conley_addInteractionLDs, variables = TREATMENTS),
    file.path(model_dir_conley100, "model_interaction_land_degradation_average_slope.rds")
  )

  interactions_LDsAdapt <- unlist(lapply(TREATMENTS, function(x) paste(x, CONTROLS_adaptation, sep = ":")))
  f_addInteractionLDs_addInteractionLDsAdapt <- update(
    f,
    as.formula(paste(". ~ . +", paste(c(interactions_LDs, interactions_LDsAdapt), collapse = " + ")))
  )
  mf_addInteractionLDs_addInteractionLDsAdapt <- fit_and_save(
    "model_interaction_land_degradation_adaptation.rds",
    f_addInteractionLDs_addInteractionLDsAdapt,
    df_ok,
    vcov_main
  )
  readr::write_rds(
    marginaleffects::avg_slopes(mf_addInteractionLDs_addInteractionLDsAdapt, variables = TREATMENTS),
    file.path(model_dir, "model_interaction_land_degradation_adaptation_average_slope.rds")
  )
  mf_conley_addInteractionLDs_addInteractionLDsAdapt <- fit_and_save(
    "model_interaction_land_degradation_adaptation.rds",
    f_addInteractionLDs_addInteractionLDsAdapt,
    df_ok,
    vcov = conley(cutoff = 100),
    out_dir = model_dir_conley100
  )
  readr::write_rds(
    marginaleffects::avg_slopes(mf_conley_addInteractionLDs_addInteractionLDsAdapt, variables = TREATMENTS),
    file.path(model_dir_conley100, "model_interaction_land_degradation_adaptation_average_slope.rds")
  )
} else {
  message("Skipped marginaleffects::avg_slopes. Set LDYG_ALLOW_AVG_SLOPES=true to recompute slow average-slope objects.")
}
