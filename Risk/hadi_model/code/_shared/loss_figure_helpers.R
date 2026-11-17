# Shared setup for production, calorie, protein, and revenue loss figures.
#
# These helpers read country-level and raster loss outputs, apply consistent
# map/legend styling, and build the bar plots used to summarize the largest
# country-level losses.

library(colorspace)
library(cowplot)
library(dplyr)
library(ggplot2)
library(grid)
library(readr)
library(sf)
library(terra)

source(file.path("code", "_shared", "map_helpers.R"))

loss_dir <- path_processed("losses")
country_loss_path <- file.path(loss_dir, "country_loss_totals.rds")

loss_paths <- c(
  production = file.path(loss_dir, "loss_production_tonnes.tif"),
  calories = file.path(loss_dir, "loss_calories_million_kcal.tif"),
  proteins = file.path(loss_dir, "loss_protein_tonnes.tif"),
  revenues = file.path(loss_dir, "loss_revenue_usd.tif")
)

input_paths <- c(
  harvest_area = file.path(loss_dir, "harvested_area_selected_crops_ha.tif"),
  attainable_yield = file.path(loss_dir, "attainable_yield_weighted_tonnes_per_ha.tif"),
  price = file.path(loss_dir, "producer_price_2010_usd_per_tonne_closest_year.tif"),
  calories = file.path(loss_dir, "food_supply_calories_2010_kcal_per_tonne.tif"),
  proteins = file.path(loss_dir, "food_supply_protein_2010_tonne_per_tonne.tif")
)

assert_loss_inputs <- function() {
  assert_files_exist(c(country_loss_path, loss_paths, input_paths))
}

read_country_loss <- function() {
  readRDS(country_loss_path) |>
    mutate(across(where(is.numeric), ~ ifelse(is.nan(.x), NA_real_, .x))) |>
    mutate(
      countrySum_loss_production_tonnes_thousand = loss_production_tonnes / 1000,
      countrySum_loss_calories_billion_kcal = loss_calorie_million_kcal / 1000,
      countrySum_loss_revenues_million_USD = loss_revenue_usd / 1e6,
      countrySum_loss_proteins_tonnes = loss_protein_tonnes
    )
}

read_loss_admin <- function() {
  read_country_boundaries()
}

read_loss_figure_data <- function(basename) {
  readRDS(path_figure_data(paste0(basename, ".rds")))
}

unwrap_figure_raster <- function(x) {
  terra::unwrap(x)
}

bar_theme <- theme(axis.title.x = element_text(size = 8))
refined_theme_barplot_2 <- theme(axis.title.x = element_text(size = 8))

refined_theme <- theme(
  legend.key.width = unit(0.3, "cm"),
  legend.key.height = unit(0.1, "cm"),
  legend.title = element_text(margin = margin(b = 2, t = 2), size = 8),
  legend.text = element_text(size = 7),
  legend.box.margin = margin(t = -25),
  plot.margin = margin(t = -10, r = -10, b = -10, l = -10, unit = "pt"),
  plot.title = element_text(hjust = 0.5, size = 10)
)

country_map_theme <- theme(
  legend.key.width = unit(0.3, "cm"),
  legend.key.height = unit(0.1, "cm"),
  legend.title = element_text(margin = margin(b = 2, t = 2), size = 8),
  legend.text = element_text(size = 7),
  legend.box.margin = margin(t = -25),
  plot.margin = margin(t = -10, r = -10, b = -10, l = -10, unit = "pt"),
  plot.title = element_text(hjust = 0.5, size = 10)
)

input_raster_theme <- theme(
  legend.key.width = unit(0.3, "cm"),
  legend.key.height = unit(0.1, "cm"),
  legend.title = element_text(margin = margin(b = 2, t = 2), size = 8),
  legend.text = element_text(size = 7),
  legend.box.margin = margin(t = -40),
  plot.margin = margin(t = -10, r = -10, b = -10, l = -10, unit = "pt"),
  plot.title = element_text(hjust = 0.5, size = 10)
)

color_palette_7 <- c("#FFFFB2", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026")

make_top_bar <- function(data, value_col, y_lab, n = 7) {
  data |>
    st_drop_geometry() |>
    select(ISO3, all_of(value_col)) |>
    filter(!is.na(.data[[value_col]])) |>
    slice_max(.data[[value_col]], n = n) |>
    ggplot(aes(x = reorder(ISO3, .data[[value_col]]), y = .data[[value_col]])) +
    geom_bar(stat = "identity", fill = "skyblue") +
    coord_flip() +
    labs(x = NULL, y = y_lab) +
    theme_bw() +
    theme(
      axis.text.y = element_text(size = 8),
      axis.text.x = element_text(size = 8),
      axis.title = element_text(size = 10),
      plot.title = element_text(size = 12, hjust = 0.5),
      legend.position = "none"
    )
}

make_top_bar_preselected <- function(data, y_lab) {
  data |>
    ggplot(aes(x = reorder(ISO3, value), y = value)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    coord_flip() +
    labs(x = NULL, y = y_lab) +
    theme_bw() +
    theme(
      axis.text.y = element_text(size = 8),
      axis.text.x = element_text(size = 8),
      axis.title = element_text(size = 10),
      plot.title = element_text(size = 12, hjust = 0.5),
      legend.position = "none"
    )
}

make_country_map <- function(country_abs, field, name, fisher = FALSE, clamp = Inf, rounding = 0) {
  data <- country_abs
  if (is.finite(clamp)) {
    data <- custom_clamp_sf_func(data, field, clamp)
  }
  plot_country_value_func_local(
    data,
    field = field,
    limitValsQuantile = 0,
    displayProj = "mollweide",
    customBreaks = fisher,
    customBreaksNumber = 7,
    customBreaksValues = NULL,
    color_palette = color_palette_7,
    name = name,
    label_value_rounding = rounding,
    nrow = 2
  )
}

make_country_map_continuous <- function(country_abs, field, name, clamp, breaks, labels, limits) {
  data <- custom_clamp_sf_func(country_abs, field, clamp)
  plot_country_value_func_local(
    data,
    field = field,
    limitValsQuantile = 0,
    displayProj = "mollweide",
    customBreaks = FALSE,
    name = name
  ) +
    colorspace::scale_fill_continuous_sequential(
      palette = "Reds",
      name = name,
      breaks = breaks,
      labels = labels,
      limits = limits,
      na.value = "transparent",
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(7, "cm"),
        barheight = unit(0.2, "cm")
      )
    )
}

plot_countrySum_relative_loss_percent <- function(country_loss_merged_sf) {
  plot_country_value_func_local(
    custom_clamp_sf_func(country_loss_merged_sf, "rel_loss_pct", 0.12),
    "rel_loss_pct",
    0,
    displayProj = "mollweide",
    name = "Relative loss (% of country production)"
  ) +
    colorspace::scale_fill_continuous_sequential(
      palette = "Reds",
      name = "Relative loss (% of country production)",
      breaks = seq(0, 0.12, 0.02),
      labels = c("0", "0.02", "0.04", "0.06", "0.08", "0.10", ">0.12"),
      limits = c(0, 0.12),
      na.value = "transparent",
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(7, "cm"),
        barheight = unit(0.2, "cm")
      )
    )
}

plot_loss_raster <- function(path_or_raster, clamp, palette, name, breaks, labels, limits, admin, transform = identity) {
  r <- if (inherits(path_or_raster, "SpatRaster")) path_or_raster else rast(path_or_raster)
  r <- transform(r)
  names(r) <- "value"
  plot_raster_func_local(
    custom_clamp_func(r, clamp),
    0,
    "%",
    "mollweide",
    admin = admin
  ) +
    colorspace::scale_fill_continuous_sequential(
      palette = palette,
      name = name,
      breaks = breaks,
      labels = labels,
      limits = limits,
      na.value = "transparent",
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(7, "cm"),
        barheight = unit(0.2, "cm")
      )
    )
}

plot_input_raster <- function(path_or_raster, clamp, palette, name, breaks, labels, limits, admin, transform = identity, zero_to_na = FALSE) {
  r <- if (inherits(path_or_raster, "SpatRaster")) path_or_raster else rast(path_or_raster)
  r <- transform(r)
  if (zero_to_na) {
    r <- terra::ifel(r == 0, NA, r)
  }
  names(r) <- "value"
  plot_raster_func_local(
    custom_clamp_func(r, clamp),
    0,
    "%",
    "mollweide",
    admin = admin
  ) +
    colorspace::scale_fill_continuous_sequential(
      palette = palette,
      name = name,
      breaks = breaks,
      labels = labels,
      limits = limits,
      na.value = "transparent",
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(7, "cm"),
        barheight = unit(0.2, "cm")
      )
    )
}
