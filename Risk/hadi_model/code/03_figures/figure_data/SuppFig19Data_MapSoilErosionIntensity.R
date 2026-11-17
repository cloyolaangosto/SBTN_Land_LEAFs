# Create figure data for Supplementary Fig. 19.
#
# This creates the base map data for the soil-erosion intensity illustration.

library(dplyr)
library(terra)

source(file.path("code", "_shared", "paths.R"))

dataset_path <- path_processed("final_dataset", "analysis_ready_data_full.rds")
yield_gap_template_path <- path_raw("crop_yield", "yield_gap_2010_percent.tif")
assert_files_exist(c(dataset_path, yield_gap_template_path))

df_ok <- readRDS(dataset_path) |>
  dplyr::select(long, lat, soilErosionDebt) |>
  tidyr::drop_na()

df_ok_sf <- sf::st_as_sf(df_ok, coords = c("long", "lat"), crs = 4326, remove = FALSE)
soilErosionDebt <- terra::rasterize(
  terra::vect(df_ok_sf),
  terra::rast(yield_gap_template_path),
  field = "soilErosionDebt",
  fun = "max"
)
names(soilErosionDebt) <- "soil_erosion_debt"

r_nonzero <- terra::clamp(soilErosionDebt, lower = 0.000001, values = FALSE)
global_median <- terra::global(r_nonzero, quantile, probs = 0.5, na.rm = TRUE)[1, 1]

if (is.na(global_median) || global_median == 0) {
  global_median <- terra::global(soilErosionDebt, "mean", na.rm = TRUE)[1, 1]
}

ratio_layer <- soilErosionDebt / global_median
names(ratio_layer) <- "ratio_to_global_median"
max_ratio <- terra::global(ratio_layer, "max", na.rm = TRUE)[1, 1]

classify_ratio <- function(threshold_center = NULL, label_name, is_extreme = FALSE) {
  if (is_extreme) {
    thresh_low <- 7.0
    if (max_ratio < 7.0) thresh_low <- 4.0
    mtx <- matrix(c(0.8, 1.2, 1, thresh_low, Inf, 2), ncol = 3, byrow = TRUE)
    lbl <- paste0("Extreme (> ", thresh_low, "x)")
  } else {
    low <- threshold_center - 0.2
    high <- threshold_center + 0.2
    mtx <- matrix(c(0.8, 1.2, 1, low, high, 2), ncol = 3, byrow = TRUE)
    lbl <- paste0(label_name, " (", low, "-", high, "x)")
  }

  r_class <- terra::classify(ratio_layer, mtx, others = NA)
  levels(r_class) <- data.frame(id = c(1, 2), label = c("Typical (0.8-1.2x)", lbl))
  r_class
}

class_stack <- c(
  classify_ratio(2.0, "Double"),
  classify_ratio(3.0, "Triple"),
  classify_ratio(4.0, "High (4x)"),
  classify_ratio(label_name = "Extreme", is_extreme = TRUE)
)
names(class_stack) <- c("double", "triple", "high_4x", "extreme")

saveRDS(
  list(
    soil_erosion_debt = terra::wrap(soilErosionDebt),
    ratio_to_global_median = terra::wrap(ratio_layer),
    class_rasters = terra::wrap(class_stack),
    input_files = c(dataset = dataset_path, yield_gap_template = yield_gap_template_path),
    summary = list(median_used = global_median, max_ratio = max_ratio)
  ),
  path_figure_data("SuppFig19Data_MapSoilErosionIntensity.rds")
)

terra::writeRaster(
  c(soilErosionDebt, ratio_layer, class_stack),
  path_figure_data("SuppFig19Data_MapSoilErosionIntensity.tif"),
  overwrite = TRUE
)
