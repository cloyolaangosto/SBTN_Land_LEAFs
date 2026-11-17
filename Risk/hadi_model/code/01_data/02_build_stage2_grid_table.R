# Build the grid-cell data table from the 10 km raster stack and local layers.
#
# This stage converts the exported raster stack to grid-cell observations,
# joins country and local covariates, constructs the final land-degradation debt
# variables used downstream.
#


library(dplyr)
library(readr)
source(file.path("code", "_shared", "analysis_paths.R"))

stage2_output <- path_intermediate("stage2_grid_table.rds")

merged_raster_input <- path_intermediate("gee_analysis_stack.tif")
country_boundaries <- path_raw("country_boundaries", "world_boundaries_fixed.gpkg")


if (!allow_heavy("LDYG_ALLOW_LOCAL_DATA_PREP")) {
  stop(
    paste(
      "Stage 2 grid-table construction is a long-running raster-to-table step.",
      "Set LDYG_ALLOW_LOCAL_DATA_PREP=true to rebuild it from the merged raster stack.",
      sep = "\n"
    ),
    call. = FALSE
  )
}

library(sf)
library(terra)

raw_dir <- path_raw()
intermediate_dir <- path_intermediate()
dir.create(intermediate_dir, recursive = TRUE, showWarnings = FALSE)

rescale_func <- function(x) {
  ((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))) * 100 + 1
}

mosaic_gee_tiles <- function(tile_files, output_tif) {
  rasters <- lapply(tile_files, terra::rast)
  merged <- do.call(terra::mosaic, c(rasters, fun = "mean"))
  terra::writeRaster(merged, output_tif, overwrite = TRUE)
  merged
}

raster_stack_to_points <- function(raster_stack, output_rds) {
  points <- terra::as.data.frame(raster_stack, xy = TRUE, na.rm = FALSE) |>
    rename(long = x, lat = y)
  write_rds(points, output_rds)
  points
}

add_country_variables <- function(df, countries_gpkg) {
  countries <- st_read(countries_gpkg, quiet = TRUE)
  df_sf <- st_as_sf(df, coords = c("long", "lat"), crs = 4326, remove = FALSE)
  joined <- st_join(df_sf, countries) |>
    st_drop_geometry()
  joined
}

# Land-degradation debts used in the manuscript analysis.
make_land_degradation_debts <- function(df) {
  df |>
    mutate(
      soilErosionDebt = Diff_Water_erosion__mean,
      soilCompactionDebt = Subsoil_compaction_susceptibility_index_mean__mean,
      soilOrganicCarbonDebt = soc_delta_ca1983_min_ca2010__mean,
      soilWaterDebt = sm_1979_1983_mean_minus_2008_2012_mean__mean,
      treeCoverDebt = Diff_treeCover__sum
    ) |>
    # Following the study definition, recode negative debt values to zero so the degradation metric captures
    # only deficits relative to the reference condition
    mutate(
      soilErosionDebt = pmax(soilErosionDebt, 0),
      soilCompactionDebt = pmax(soilCompactionDebt, 0),
      soilOrganicCarbonDebt = pmax(soilOrganicCarbonDebt, 0),
      soilWaterDebt = pmax(soilWaterDebt, 0),
      treeCoverDebt = pmax(treeCoverDebt, 0)
    )
}

add_final_prep_variables <- function(df) {
  df |>
    mutate(
      pr_end_sum_mean__mean_sq = pr_end_sum_mean__mean^2,
      tmmn_end_mean_mean__mean_sq = tmmn_end_mean_mean__mean^2,
      tmmx_end_mean_mean__mean_sq = tmmx_end_mean_mean__mean^2,
      tmmn_mean_mean__mean_delta_sq = tmmn_mean_mean__mean_delta^2,
      tmmx_mean_mean__mean_delta_sq = tmmx_mean_mean__mean_delta^2,
      pr_sum_mean__mean_delta_sq = pr_sum_mean__mean_delta^2,
      wc_tavg__mean_sq = wc_tavg__mean^2,
      wc_prec__mean_sq = wc_prec__mean^2,
      wc_meanTempWarmestQuarter__mean_sq = wc_meanTempWarmestQuarter__mean^2,
      wc_maxTempWarmestMonth__mean_sq = wc_maxTempWarmestMonth__mean^2,
      wc_precWettestQuarter__mean_sq = wc_precWettestQuarter__mean^2
    ) |>
    mutate(
      soil_type__mode = as.factor(soil_type__mode),
      dominantFieldSizeCategory = as.factor(dominantFieldSizeCategory__mode)
    )
}

add_rescaled_logs <- function(df) {
  # Following the study specification, log-transform for elasticity-style coefficient interpretation
  df |>
    mutate(across(where(is.numeric), list(rescaled = ~ rescale_func(.)), .names = "{col}__{fn}")) |>
    mutate(across(ends_with("__rescaled"), list(log = ~ log(.)), .names = "{col}__{fn}"))
}

assert_files_exist(c(merged_raster_input, country_boundaries))

# Build the Stage 2 grid table from the shareable merged raster stack.
merged_raster <- terra::rast(merged_raster_input)
df <- raster_stack_to_points(
  merged_raster,
  file.path(intermediate_dir, "gee_analysis_stack_points.rds")
)
df <- add_country_variables(
  df,
  country_boundaries
)

df <- make_land_degradation_debts(df)
df <- add_final_prep_variables(df)
df <- add_rescaled_logs(df)

df_ok <- df |>
  filter(!is.na(AllCrops2010YieldGapPercent)) |>
  mutate(lat = lat, long = long)

write_rds(df_ok, stage2_output)
message("Created Stage 2 analytical dataset: ", stage2_output)

