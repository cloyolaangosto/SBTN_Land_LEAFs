# Build the final full analysis table used by model and figure scripts.
#
#


library(dplyr)
library(readr)
source(file.path("code", "_shared", "analysis_paths.R"))
source(file.path("code", "01_data", "dataset_contract.R"))

stage2_input <- path_intermediate("stage2_grid_table.rds")
final_dataset <- historical_final_dataset_path()
final_dataset_vars_only <- analysis_ready_dataset_path()
coarser_50km <- path_intermediate("coarser_resolution", "loglog", "analysis_ready_data_50km.rds")
coarser_100km <- path_intermediate("coarser_resolution", "loglog", "analysis_ready_data_100km.rds")
actual_yield_raster_path <- path_raw("actual_yields", "actual_yield_2010_weighted.tif")
attainable_yield_raster_path <- path_raw("attainable_yields", "attainable_yield_2010_weighted.tif")
dominant_crop_lookup_path <- path_processed("final_dataset", "dominant_crop_type.csv")


if (!allow_heavy("LDYG_ALLOW_SERVER_DATA_PREP")) {
  stop(
    paste(
      "Final analysis-table construction extracts additional rasters to the grid table and can be slow.",
      "Set LDYG_ALLOW_SERVER_DATA_PREP=true to rebuild the full analysis table.",
      sep = "\n"
    ),
    call. = FALSE
  )
}

assert_files_exist(c(stage2_input, actual_yield_raster_path, attainable_yield_raster_path))

library(sf)
library(terra)

cf_data_dir <- path_processed("final_dataset")
raw_dir <- path_raw()
dir.create(cf_data_dir, recursive = TRUE, showWarnings = FALSE)

rescale_func <- function(x) {
  min_x <- min(x, na.rm = TRUE)
  x - min_x + 1
}

extract_raster_to_df <- function(df_ok, raster_layer) {
  df_sf <- df_ok |>
    st_as_sf(coords = c("long", "lat"), crs = 4326, remove = FALSE)
  df_tr <- terra::extract(
    x = raster_layer,
    y = vect(df_sf),
    method = "simple",
    bind = TRUE
  )
  st_as_sf(df_tr) |>
    st_drop_geometry()
}

df_ok <- read_rds(stage2_input)
df_final <- df_ok

actual_yield <- rast(actual_yield_raster_path)
names(actual_yield) <- "actualYieldGerberRay"
df_final <- extract_raster_to_df(df_final, actual_yield) |>
  mutate(
    actualYieldGerberRay__rescaled = rescale_func(actualYieldGerberRay),
    actualYieldGerberRay__rescaled__log = log(actualYieldGerberRay__rescaled)
  )

attainable_yield <- rast(attainable_yield_raster_path)
names(attainable_yield) <- "attainableYieldGerberRay"
df_final <- extract_raster_to_df(df_final, attainable_yield) |>
  mutate(
    attainableYieldGerberRay__rescaled = rescale_func(attainableYieldGerberRay),
    attainableYieldGerberRay__rescaled__log = log(attainableYieldGerberRay__rescaled)
  )

if (!all(crop_type_dummy_vars %in% names(df_final))) {
  if (!file.exists(dominant_crop_lookup_path)) {
    stop(
      "Missing derived Ray/Gerber dominant-crop lookup: ",
      dominant_crop_lookup_path,
      "\nThe derived lookup is required because per-crop source rasters are not distributed with this package.",
      call. = FALSE
    )
  }
  crop_lookup <- read_csv(dominant_crop_lookup_path, show_col_types = FALSE)
  df_final <- df_final |>
    left_join(crop_lookup, by = c("long", "lat"))
}

write_rds(df_final, final_dataset)
message("Created final full analysis dataset: ", final_dataset)

validation <- validate_analysis_ready_columns(df_final)
if (!validation$ok) {
  stop(
    "Final dataset is missing required vars-only analysis columns: ",
    paste(unique(c(validation$missing_keep, validation$missing_regression, validation$missing_causal_forest)), collapse = ", "),
    call. = FALSE
  )
}

df_final_vars_only <- df_final |>
  select(all_of(analysis_ready_keep_vars))
write_rds(df_final_vars_only, final_dataset_vars_only)
message("Created variables-only analysis-ready dataset: ", final_dataset_vars_only)

