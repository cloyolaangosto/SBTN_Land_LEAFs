# Create figure data for Main Fig. 1 and Supplementary Fig. 1.

library(dplyr)
library(readr)
library(terra)

source(file.path("code", "_shared", "paths.R"))
source(file.path("code", "_shared", "map_helpers.R"))

dataset_path <- path_processed("final_dataset", "analysis_ready_data_full.rds")
yield_gap_template_path <- path_raw("crop_yield", "yield_gap_2010_percent.tif")
model_var_path <- path_processed(
  "models",
  "main_10km",
  "model_spec_variables.rds"
)

assert_files_exist(c(dataset_path, yield_gap_template_path, model_var_path))

df_ok <- readRDS(dataset_path)
model_vars <- readRDS(model_var_path)
reg_varNames_flat <- unname(unlist(model_vars))
reg_varNames_flat_notLog <- stringr::str_replace_all(reg_varNames_flat, "__log", "")
reg_varNames_flat_notLog_notScaled <- stringr::str_replace_all(reg_varNames_flat, "__rescaled__log", "")

needed_vars <- unique(c(
  "long", "lat",
  "soilErosionDebt",
  "soilCompactionDebt",
  "soilOrganicCarbonDebt",
  "soilWaterDebt",
  "treeCoverDebt",
  "AllCrops2010YieldGapPercent",
  reg_varNames_flat,
  reg_varNames_flat_notLog,
  reg_varNames_flat_notLog_notScaled
))
needed_vars <- intersect(needed_vars, names(df_ok))

df_ok_reg <- df_ok |>
  dplyr::select(dplyr::all_of(needed_vars)) |>
  tidyr::drop_na()

df_ok_reg_sf <- sf::st_as_sf(df_ok_reg, coords = c("long", "lat"), crs = 4326, remove = FALSE)
df_ok_reg_tr <- terra::vect(df_ok_reg_sf)
yieldGap_tr <- terra::rast(yield_gap_template_path)

rasterize_func <- function(field) {
  terra::rasterize(
    x = df_ok_reg_tr,
    y = yieldGap_tr,
    field = field,
    fun = "max"
  )
}

fig1_stack <- c(
  rasterize_func("soilErosionDebt"),
  rasterize_func("soilCompactionDebt"),
  rasterize_func("soilOrganicCarbonDebt"),
  rasterize_func("soilWaterDebt"),
  rasterize_func("treeCoverDebt"),
  rasterize_func("AllCrops2010YieldGapPercent")
)

names(fig1_stack) <- c(
  "soilErosionDebt",
  "soilCompactionDebt",
  "soilOrganicCarbonDebt",
  "soilWaterDebt",
  "treeCoverDebt",
  "AllCrops2010YieldGapPercent"
)

fig1_raster_quantiles <- dplyr::bind_rows(
  quantile_tbl(fig1_stack[["soilErosionDebt"]], "soilErosionDebt"),
  quantile_tbl(fig1_stack[["soilCompactionDebt"]], "soilCompactionDebt"),
  quantile_tbl(fig1_stack[["soilOrganicCarbonDebt"]], "soilOrganicCarbonDebt"),
  quantile_tbl(fig1_stack[["soilWaterDebt"]], "soilWaterDebt"),
  quantile_tbl(fig1_stack[["treeCoverDebt"]], "treeCoverDebt"),
  quantile_tbl(fig1_stack[["AllCrops2010YieldGapPercent"]], "AllCrops2010YieldGapPercent")
)

fig1_figure_data <- list(
  rasters = terra::wrap(fig1_stack),
  input_files = c(dataset = dataset_path, yield_gap_template = yield_gap_template_path, model_vars = model_var_path),
  n_input_rows = nrow(df_ok),
  n_complete_case_rows = nrow(df_ok_reg),
  quantiles = fig1_raster_quantiles,
  map_maxcell = Sys.getenv("LDYG_MAP_MAXCELL", unset = "200000")
)

saveRDS(fig1_figure_data, path_figure_data("Fig1Data_MapLandDegradationYieldGaps.rds"))
saveRDS(fig1_figure_data, path_figure_data("SuppFig1Data_MapLandDegradationYieldGapFisher7.rds"))

terra::writeRaster(
  fig1_stack,
  path_figure_data("Fig1Data_MapLandDegradationYieldGaps.tif"),
  overwrite = TRUE
)
terra::writeRaster(
  fig1_stack,
  path_figure_data("SuppFig1Data_MapLandDegradationYieldGapFisher7.tif"),
  overwrite = TRUE
)

readr::write_csv(fig1_raster_quantiles, path_figure_data("Fig1Data_MapLandDegradationYieldGapsRasterQuantiles.csv"))
readr::write_csv(fig1_raster_quantiles, path_figure_data("SuppFig1Data_MapLandDegradationYieldGapFisher7RasterQuantiles.csv"))
