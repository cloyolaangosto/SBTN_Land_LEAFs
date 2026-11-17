# Shared setup for causal-forest CATE map figure scripts.
#
# These helpers read the saved post-fit CATE tables/rasters, join them to the
# analysis grid or country boundaries, and apply the common map styling used for
# the overall and land-degradation-process-specific effect maps.

library(colorspace)
library(cowplot)
library(dplyr)
library(ggplot2)
library(grid)
library(readr)
library(sf)
library(terra)

source(file.path("code", "_shared", "map_helpers.R"))

cf_dir <- path_processed("causal_forest")
cf_raster_dir <- file.path(cf_dir, "rasterized")

cate_sum_path <- file.path(cf_raster_dir, "cate_sum_nonneg.tif")
yield_gap_path <- path_raw("crop_yield", "yield_gap_2010_percent.tif")

individual_paths <- c(
  EROSION = file.path(cf_raster_dir, "cate_erosion_nonneg.tif"),
  COMPACTION = file.path(cf_raster_dir, "cate_compaction_nonneg.tif"),
  SOC = file.path(cf_raster_dir, "cate_soc_nonneg.tif"),
  SOILWATER = file.path(cf_raster_dir, "cate_soil_water_nonneg.tif"),
  TREECOVER = file.path(cf_raster_dir, "cate_tree_cover_nonneg.tif")
)

tuned_inputs <- c(
  SuppFig13_MapCateErosionTuned = file.path(cf_dir, "cate_erosion_tuned.rds"),
  SuppFig14_MapCateErosionSeparateModels = file.path(cf_dir, "cate_erosion_separate_models.rds")
)

assert_cate_base_inputs <- function(include_yield_gap = FALSE) {
  paths <- c(cate_sum_path, individual_paths)
  if (include_yield_gap) {
    paths <- c(paths, yield_gap_path)
  }
  assert_files_exist(paths)
}

read_cate_admin <- function() {
  read_country_boundaries()
}

read_cate_sum <- function() {
  r <- rast(cate_sum_path)
  names(r) <- "CATE_sum"
  r
}

read_yield_gap <- function() {
  r <- rast(yield_gap_path)
  names(r) <- "yield_gap"
  r
}

read_cate_figure_data <- function(basename) {
  readRDS(path_figure_data(paste0(basename, ".rds")))
}

unwrap_figure_raster <- function(x) {
  terra::unwrap(x)
}

refined_theme_sum <- theme(
  legend.key.width = unit(0.1, "cm"),
  legend.key.height = unit(0.2, "cm"),
  legend.title = element_text(margin = margin(b = 2, t = 2), size = 10),
  legend.text = element_text(size = 8),
  legend.box.margin = margin(t = -50),
  plot.margin = margin(t = -20, r = -20, b = -20, l = -20, unit = "pt"),
  plot.title = element_text(hjust = 0.5, size = 10)
)

refined_theme_fisher <- theme(
  legend.key.width = unit(0.4, "cm"),
  legend.key.height = unit(0.1, "cm"),
  legend.title = element_text(size = 10),
  legend.text = element_text(size = 8),
  legend.box.margin = margin(t = -50),
  plot.margin = margin(t = -20, r = -20, b = -20, l = -20, unit = "pt"),
  plot.title = element_text(hjust = 0.5, size = 10)
)

color_palette_7 <- c("#FFFFB2", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026")

plot_cate_relative_change <- function(cate_sum, admin) {
  plot_raster_func_local(
    custom_clamp_func(cate_sum, 0.4),
    0,
    "%",
    "mollweide",
    admin = admin
  ) +
    colorspace::scale_fill_continuous_sequential(
      palette = "Reds",
      name = "Relative change (%) in yield gaps associated with a 1% increase in land degradation",
      breaks = seq(0, 0.4, 0.05),
      labels = c("0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", ">0.4"),
      na.value = "transparent",
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(10, "cm"),
        barheight = unit(0.2, "cm")
      )
    )
}

plot_cate_absolute_pp_change <- function(cate_sum, yield_gap, admin) {
  abs_change_yield_gap_pp <- (cate_sum / 100) * yield_gap
  names(abs_change_yield_gap_pp) <- "absolute_pp"
  plot_raster_func_local(
    custom_clamp_func(abs_change_yield_gap_pp, 0.12),
    0,
    "%",
    "mollweide",
    admin = admin
  ) +
    colorspace::scale_fill_continuous_sequential(
      palette = "Reds",
      name = "Absolute change (pp) in yield gaps associated with a 1% increase in land degradation",
      breaks = seq(0, 0.12, 0.02),
      labels = c("0", "0.02", "0.04", "0.06", "0.08", "0.10", ">0.12"),
      limits = c(0, 0.12),
      na.value = "transparent",
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(10, "cm"),
        barheight = unit(0.2, "cm")
      )
    )
}

plot_cate_sum_fisher <- function(cate_sum, admin) {
  plot_raster_func_local(
    cate_sum,
    0,
    "%",
    "mollweide",
    admin = admin,
    customBreaks = TRUE,
    customBreaksNumber = 7,
    customBreaksValues = NULL,
    samp_prop = 1,
    color_palette = color_palette_7,
    name = "Change in yield gaps (%)",
    label_value_rounding = 2,
    nrow = 1
  )
}

plot_one_cate <- function(path_or_raster, clamp, palette, breaks, labels, admin) {
  r <- if (inherits(path_or_raster, "SpatRaster")) path_or_raster else rast(path_or_raster)
  names(r) <- "CATE"
  plot_raster_func_local(
    custom_clamp_func(r, clamp),
    0,
    "%",
    "mollweide",
    admin = admin
  ) +
    colorspace::scale_fill_continuous_sequential(
      palette = palette,
      name = "Change in yield gaps (%)",
      breaks = breaks,
      labels = labels,
      na.value = "transparent",
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(10, "cm"),
        barheight = unit(0.2, "cm")
      )
    )
}

individual_specs <- list(
  EROSION = list(path = individual_paths[["EROSION"]], clamp = 0.2, palette = "Reds 2", breaks = seq(0, 0.2, 0.05), labels = c("0", "0.05", "0.1", "0.15", ">0.2"), filename = "SuppFig8_MapCateErosion"),
  COMPACTION = list(path = individual_paths[["COMPACTION"]], clamp = 0.15, palette = "Reds", breaks = seq(0, 0.15, 0.03), labels = c("0", "0.03", "0.06", "0.09", "0.12", ">0.15"), filename = "SuppFig9_MapCateCompaction"),
  SOC = list(path = individual_paths[["SOC"]], clamp = 0.05, palette = "Peach", breaks = seq(0, 0.05, 0.01), labels = c("0", "0.01", "0.02", "0.03", "0.04", ">0.05"), filename = "SuppFig10_MapCateSoc"),
  SOILWATER = list(path = individual_paths[["SOILWATER"]], clamp = 0.075, palette = "Blues", breaks = seq(0, 0.075, 0.025), labels = c("0", "0.025", "0.05", ">0.075"), filename = "SuppFig11_MapCateSoilWater"),
  TREECOVER = list(path = individual_paths[["TREECOVER"]], clamp = 0.15, palette = "Greens", breaks = seq(0, 0.15, 0.03), labels = c("0", "0.03", "0.06", "0.09", "0.12", ">0.15"), filename = "SuppFig12_MapCateTreeCover")
)

plot_individual_cate_by_id <- function(id, admin) {
  spec <- individual_specs[[id]]
  plot_one_cate(spec$path, spec$clamp, spec$palette, spec$breaks, spec$labels, admin)
}

plot_individual_cate_raster_by_id <- function(id, cate, admin) {
  spec <- individual_specs[[id]]
  plot_one_cate(cate, spec$clamp, spec$palette, spec$breaks, spec$labels, admin)
}

plot_other_cf_experiment <- function(df_cate, admin, yield_gap) {
  df_cate <- df_cate |>
    mutate(CATE = as.numeric(CATE))
  df_sf <- sf::st_as_sf(df_cate, coords = c("long", "lat"), crs = 4326, remove = FALSE)
  r <- terra::rasterize(terra::vect(df_sf), yield_gap, field = "CATE", fun = "max")
  r <- terra::ifel(r < 0, 0, r)
  plot_raster_func_local(
    custom_clamp_func(r, 0.2),
    0,
    "%",
    "mollweide",
    admin = admin
  ) +
    colorspace::scale_fill_continuous_sequential(
      palette = "Reds 2",
      name = "Change in yield gaps (%)",
      breaks = seq(0, 0.2, 0.05),
      labels = c("0", "0.05", "0.1", "0.15", ">0.2"),
      na.value = "transparent",
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(10, "cm"),
        barheight = unit(0.2, "cm")
      )
    )
}
