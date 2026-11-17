# Fit the causal forests used to estimate heterogeneous yield-gap responses.
#
# The manuscript maps conditional average treatment effects (CATEs) for five
# land-degradation debt variables using the variables-only analysis dataset:
#   - variables-only analysis dataset:
#     analysis_ready_data_regression_vars.rds



library(dplyr)
library(readr)
library(tibble)

source(file.path("code", "_shared", "analysis_paths.R"))
source(file.path("code", "01_data", "dataset_contract.R"))

dataset_path <- analysis_ready_dataset_path()
yield_gap_template_path <- path_raw("crop_yield", "yield_gap_2010_percent.tif")
cf_dir <- path_processed("causal_forest")
cf_raster_dir <- file.path(cf_dir, "rasterized")

input_dependencies <- c(dataset_path)

cf_raster_outputs <- c(
  file.path(cf_raster_dir, "cate_sum_nonneg.tif"),
  file.path(cf_raster_dir, "cate_erosion_nonneg.tif"),
  file.path(cf_raster_dir, "cate_compaction_nonneg.tif"),
  file.path(cf_raster_dir, "cate_soc_nonneg.tif"),
  file.path(cf_raster_dir, "cate_soil_water_nonneg.tif"),
  file.path(cf_raster_dir, "cate_tree_cover_nonneg.tif")
)

planned_cf_fit_outputs <- c(
  file.path(cf_dir, "causal_forest_erosion_crop_type_controls.rds"),
  file.path(cf_dir, "causal_forest_compaction_crop_type_controls.rds"),
  file.path(cf_dir, "causal_forest_soc_crop_type_controls.rds"),
  file.path(cf_dir, "causal_forest_soil_water_crop_type_controls.rds"),
  file.path(cf_dir, "causal_forest_tree_cover_crop_type_controls.rds"),
  file.path(cf_dir, "cate_erosion_crop_type_controls.rds"),
  file.path(cf_dir, "cate_compaction_crop_type_controls.rds"),
  file.path(cf_dir, "cate_soc_crop_type_controls.rds"),
  file.path(cf_dir, "cate_soil_water_crop_type_controls.rds"),
  file.path(cf_dir, "cate_tree_cover_crop_type_controls.rds")
)


if (!allow_heavy("LDYG_ALLOW_CAUSAL_FOREST_FIT") && !allow_heavy("LDYG_ALLOW_CATE_RASTERIZE")) {
  stop(
    paste(
      "Causal forest fitting and CATE rasterization are long-running stages.",
      "Set LDYG_ALLOW_CAUSAL_FOREST_FIT=true to fit forests, or LDYG_ALLOW_CATE_RASTERIZE=true to rasterize saved CATE tables.",
      sep = "\n"
    ),
    call. = FALSE
  )
}

if (allow_heavy("LDYG_ALLOW_CAUSAL_FOREST_FIT")) {
  stop_unless_heavy_allowed(
    "LDYG_ALLOW_CAUSAL_FOREST_FIT",
    "Causal forest fitting is unlikely to be practical on a standard laptop; in our runs a single fit could take up to about 8 hours and use roughly 20-30 GB RAM."
  )
  assert_files_exist(input_dependencies)

  library(grf)
  library(data.table)

config <- list(
  binaryTreatment = FALSE,
  linear_linear_model = TRUE,
  augmentControlsForSOFA = FALSE,
  maximizeNObs = TRUE,
  testSalinityDebt = FALSE,
  addRayGerberCropTypes = TRUE
)

df_ok <- read_rds(dataset_path) |>
  as_tibble() |>
  mutate(row_id = row_number())

OUTCOME <- "AllCrops2010YieldGapPercent"

TREATMENTS <- c(
  "soilErosionDebt",
  "soilCompactionDebt",
  "soilOrganicCarbonDebt",
  "soilWaterDebt",
  "treeCoverDebt"
)

CONTROLS_environment_timeDelta_climateTerra <- c(
  "pr_sum_mean__mean_delta",
  "tmmn_mean_mean__mean_delta",
  "tmmx_mean_mean__mean_delta",
  "srad_mean_mean__mean_delta"
)

CONTROLS_environment_climateTerra <- c(
  "pr_end_sum_mean__mean",
  "tmmn_end_mean_mean__mean",
  "tmmx_end_mean_mean__mean",
  "srad_end_mean_mean__mean"
)

CONTROLS_environment_terrain <- c(
  "gmted_elevation__mean",
  "gmted_slope__mean",
  "geomorpho90m_tri__mean"
)

CONTROLS_environment_soilType <- "soil_type__mode"

CONTROLS_environment_soilGrid <- c(
  "clay__mean",
  "sand__mean",
  "silt__mean"
)

CONTROLS_environment_climateWc <- c(
  "wc_srad__mean",
  "wc_tavg__mean",
  "wc_prec__mean",
  "wc_meanTempWarmestQuarter__mean",
  "wc_maxTempWarmestMonth__mean",
  "wc_precWettestQuarter__mean",
  "aridity__mean"
)

CONTROLS_adaptation <- c(
  "nfer_crop_nh4_end_mean__mean",
  "nfer_crop_no3_end_mean__mean",
  "nmanure_app_crop_end_mean__mean",
  "ndep_noy_end_mean__mean",
  "ndep_nhx_end_mean__mean",
  "phosphorus_OlsenP_kgha1_World_Aug2022_ver__mean",
  "pesticide_APR_sum_2015_H__mean",
  "irrigation_AEI_ha_end__mean"
)

CONTROLS_socioeconomic <- c(
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

df_ok_noNA <- df_ok |>
  dplyr::select(all_of(c(
    "row_id", "long", "lat",
    OUTCOME,
    TREATMENTS,
    "attainableYieldGerberRay",
    "actualYieldGerberRay",
    crop_type_dummy_vars,
    CONTROLS_environment_climateWc,
    CONTROLS_environment_climateTerra,
    CONTROLS_environment_timeDelta_climateTerra,
    CONTROLS_environment_terrain,
    CONTROLS_environment_soilType,
    CONTROLS_environment_soilGrid,
    CONTROLS_adaptation,
    CONTROLS_socioeconomic
  ))) |>
  tidyr::drop_na() |>
  mutate(across(all_of(crop_type_dummy_vars), as.integer))

write_rds(
  df_ok_noNA,
  path_processed("final_dataset", "analysis_ready_data_linear_scale.rds")
)

soil_type_HOT <- model.matrix(~ soil_type__mode - 1, data = df_ok_noNA)

X_base <- df_ok_noNA |>
  dplyr::select(all_of(c(
    CONTROLS_environment_climateWc,
    CONTROLS_environment_climateTerra,
    CONTROLS_environment_timeDelta_climateTerra,
    CONTROLS_environment_terrain,
    CONTROLS_environment_soilGrid,
    CONTROLS_adaptation,
    CONTROLS_socioeconomic
  ))) |>
  as.matrix() |>
  bind_cols(soil_type_HOT)

Y <- df_ok_noNA |>
  dplyr::select(all_of(OUTCOME)) |>
  pull()

run_cf_func_customXbase <- function(D_var,
                                    custom_X_base = X_base,
                                    out_cf_filename,
                                    out_df_CATE_filename,
                                    tune.parameters = "none",
                                    num.trees = 2000,
                                    min.node.size = 5,
                                    imbalance.penalty = 0,
                                    num.threads = 4) {
  D <- df_ok_noNA |>
    select(all_of(D_var)) |>
    pull()

  otherLDs_var <- setdiff(TREATMENTS, D_var)
  X_otherLDs <- df_ok_noNA |>
    select(all_of(otherLDs_var)) |>
    as.matrix()
  X <- custom_X_base %>% bind_cols(X_otherLDs)

  set.seed(42)
  cf <- causal_forest(
    X = X,
    Y = Y,
    W = D,
    num.threads = num.threads,
    tune.parameters = tune.parameters,
    num.trees = num.trees,
    min.node.size = min.node.size,
    imbalance.penalty = imbalance.penalty
  )

  prediction <- predict(cf)

  write_rds(cf, file.path(cf_dir, paste0(out_cf_filename, ".rds")))

  df_CATE <- df_ok_noNA |>
    select(long, lat) |>
    mutate(
      CATE = prediction$predictions
    )

  data.table::fwrite(
    df_CATE,
    file.path(cf_dir, paste0(out_df_CATE_filename, ".csv")),
    nThread = 4,
    verbose = TRUE,
    showProgress = TRUE
  )
  write_rds(df_CATE, file.path(cf_dir, paste0(out_df_CATE_filename, ".rds")))
}

X_base_with_crop_type_controls <- X_base %>%
  bind_cols(
    df_ok_noNA |>
      dplyr::select(all_of(crop_type_dummy_vars)) |>
      as.matrix()
  )

  # CATE-by-crop inspection requires crop-type-control forests for all five land
  # degradation processes, so those fitting calls are made explicit here.
  run_cf_func_customXbase(
    D_var = "soilErosionDebt",
    custom_X_base = X_base_with_crop_type_controls,
    out_cf_filename = "causal_forest_erosion_crop_type_controls",
    out_df_CATE_filename = "cate_erosion_crop_type_controls"
  )
  run_cf_func_customXbase(
    D_var = "soilCompactionDebt",
    custom_X_base = X_base_with_crop_type_controls,
    out_cf_filename = "causal_forest_compaction_crop_type_controls",
    out_df_CATE_filename = "cate_compaction_crop_type_controls"
  )
  run_cf_func_customXbase(
    D_var = "soilOrganicCarbonDebt",
    custom_X_base = X_base_with_crop_type_controls,
    out_cf_filename = "causal_forest_soc_crop_type_controls",
    out_df_CATE_filename = "cate_soc_crop_type_controls"
  )
  run_cf_func_customXbase(
    D_var = "soilWaterDebt",
    custom_X_base = X_base_with_crop_type_controls,
    out_cf_filename = "causal_forest_soil_water_crop_type_controls",
    out_df_CATE_filename = "cate_soil_water_crop_type_controls"
  )
  run_cf_func_customXbase(
    D_var = "treeCoverDebt",
    custom_X_base = X_base_with_crop_type_controls,
    out_cf_filename = "causal_forest_tree_cover_crop_type_controls",
    out_df_CATE_filename = "cate_tree_cover_crop_type_controls"
  )
}

if (!allow_heavy("LDYG_ALLOW_CATE_RASTERIZE")) {
  stop("Set LDYG_ALLOW_CATE_RASTERIZE=true to rasterize saved CATE tables.", call. = FALSE)
}

stop_unless_heavy_allowed(
  "LDYG_ALLOW_CATE_RASTERIZE",
  "CATE rasterization is a post-fit step and can be run from saved CATE tables without fitting forests."
)

# Rasterize CATE tables so downstream map and loss scripts can use the same
# spatial grid as the analysis data.
library(raster)
library(sf)
library(terra)

dir.create(cf_raster_dir, recursive = TRUE, showWarnings = FALSE)

assert_files_exist(c(yield_gap_template_path, planned_cf_fit_outputs[6:10]))

yieldGap_rst <- raster::raster(yield_gap_template_path)
wopt_options <- list(
  gdal = c("COMPRESS=DEFLATE", "NUM_THREADS=10", "BLOCKXSIZE=512", "BLOCKYSIZE=512"),
  datatype = "FLT4S",
  NAflag = -3.4e+38
)

read_cate <- function(name, cate_name) {
  read_rds(file.path(cf_dir, paste0(name, ".rds"))) |>
    rename(!!cate_name := CATE)
}

# The total CATE combines the five process-specific predictions
df_CATE_merged <- read_cate("cate_erosion_crop_type_controls", "CATE_EROSION") |>
  inner_join(read_cate("cate_compaction_crop_type_controls", "CATE_COMPACTION"), by = c("long", "lat")) |>
  inner_join(read_cate("cate_soc_crop_type_controls", "CATE_SOC"), by = c("long", "lat")) |>
  inner_join(read_cate("cate_soil_water_crop_type_controls", "CATE_SOILWATER"), by = c("long", "lat")) |>
  inner_join(read_cate("cate_tree_cover_crop_type_controls", "CATE_TREECOVER"), by = c("long", "lat"))

df_CATE_merged_sf <- sf::st_as_sf(df_CATE_merged, coords = c("long", "lat"), crs = 4326, remove = FALSE)

rasterize_field <- function(field) {
  terra::rast(
    raster::rasterize(
      x = df_CATE_merged_sf,
      y = yieldGap_rst,
      field = field,
      fun = "max"
    )
  )
}

CATE_EROSION_rst <- rasterize_field("CATE_EROSION")
CATE_COMPACTION_rst <- rasterize_field("CATE_COMPACTION")
CATE_SOC_rst <- rasterize_field("CATE_SOC")
CATE_SOILWATER_rst <- rasterize_field("CATE_SOILWATER")
CATE_TREECOVER_rst <- rasterize_field("CATE_TREECOVER")

# For visualization and aggregation, retain only positive estimated CATEs to focus the analysis on
# locations where degradation is associated with higher yield gaps
CATE_EROSION_rst_nonneg <- terra::ifel(CATE_EROSION_rst < 0, 0, CATE_EROSION_rst)
CATE_COMPACTION_rst_nonneg <- terra::ifel(CATE_COMPACTION_rst < 0, 0, CATE_COMPACTION_rst)
CATE_SOC_rst_nonneg <- terra::ifel(CATE_SOC_rst < 0, 0, CATE_SOC_rst)
CATE_SOILWATER_rst_nonneg <- terra::ifel(CATE_SOILWATER_rst < 0, 0, CATE_SOILWATER_rst)
CATE_TREECOVER_rst_nonneg <- terra::ifel(CATE_TREECOVER_rst < 0, 0, CATE_TREECOVER_rst)

terra::writeRaster(CATE_EROSION_rst_nonneg, file.path(cf_raster_dir, "cate_erosion_nonneg.tif"), overwrite = TRUE, wopt = wopt_options)
terra::writeRaster(CATE_COMPACTION_rst_nonneg, file.path(cf_raster_dir, "cate_compaction_nonneg.tif"), overwrite = TRUE, wopt = wopt_options)
terra::writeRaster(CATE_SOC_rst_nonneg, file.path(cf_raster_dir, "cate_soc_nonneg.tif"), overwrite = TRUE, wopt = wopt_options)
terra::writeRaster(CATE_SOILWATER_rst_nonneg, file.path(cf_raster_dir, "cate_soil_water_nonneg.tif"), overwrite = TRUE, wopt = wopt_options)
terra::writeRaster(CATE_TREECOVER_rst_nonneg, file.path(cf_raster_dir, "cate_tree_cover_nonneg.tif"), overwrite = TRUE, wopt = wopt_options)

CATE_rst_nonneg_sum <- CATE_EROSION_rst_nonneg + CATE_COMPACTION_rst_nonneg + CATE_SOC_rst_nonneg + CATE_SOILWATER_rst_nonneg + CATE_TREECOVER_rst_nonneg
terra::writeRaster(CATE_rst_nonneg_sum, file.path(cf_raster_dir, "cate_sum_nonneg.tif"), overwrite = TRUE, wopt = wopt_options)



