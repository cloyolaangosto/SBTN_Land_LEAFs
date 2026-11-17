# Shared map-rendering helpers for land-degradation, CATE, and loss figures.
#
# Continuous and Fisher-class maps use common clamping, classification, and
# administrative-boundary styling so figure panels remain comparable. Set
# LDYG_MAP_MAXCELL to a larger value or "Inf" for dense final map rendering.

library(classInt)
library(dplyr)
library(ggplot2)
library(sf)
library(terra)
library(tidyterra)

source(file.path("code", "_shared", "paths.R"))

country_boundary_path <- file.path(
  data_root,
  "raw",
  "country_boundaries",
  "world_boundaries_fixed.gpkg"
)

read_country_boundaries <- function() {
  assert_files_exist(country_boundary_path)
  sf::st_read(country_boundary_path, quiet = TRUE) |>
    dplyr::filter(NAME != "Antarctica")
}

map_maxcell <- function(r) {
  value <- Sys.getenv("LDYG_MAP_MAXCELL", unset = "200000")
  if (tolower(value) %in% c("inf", "infinite", "all")) {
    return(terra::ncell(r))
  }
  as.integer(value)
}

custom_clamp_func <- function(x, upper) {
  terra::ifel(x > upper, upper, x)
}

custom_clamp_sf_func <- function(x, field, upper) {
  x |>
    mutate("{field}" := if_else(.data[[field]] > upper, upper, .data[[field]]))
}

classify_raster_fisher <- function(r, n = 7, samp_prop = 1) {
  vals <- terra::values(r, mat = FALSE, na.rm = TRUE)
  large_n_setting <- Sys.getenv("LDYG_CLASSINT_LARGEN", unset = "3000")
  large_n <- if (tolower(large_n_setting) %in% c("inf", "infinite", "all")) {
    length(vals) + 1L
  } else {
    as.integer(large_n_setting)
  }
  breaks <- classInt::classIntervals(
    vals,
    n = n,
    style = "fisher",
    samp_prop = samp_prop,
    largeN = large_n
  )$brks
  class_matrix <- cbind(
    breaks[-length(breaks)],
    breaks[-1],
    seq_len(length(breaks) - 1)
  )
  classified <- terra::classify(r, class_matrix, include.lowest = TRUE, right = TRUE)
  classified <- terra::ifel(classified == 0, NA, classified)
  list(raster = as.factor(classified), breaks = breaks)
}

plot_raster_func_local <- function(mySpatRast,
                                   limitValsQuantile,
                                   legendTitle = "CATE (%)",
                                   displayProj = "planar",
                                   admin = NULL,
                                   customBreaks = FALSE,
                                   customBreaksNumber = 5,
                                   customBreaksValues = NULL,
                                   samp_prop = 1,
                                   color_palette,
                                   name,
                                   label_value_rounding,
                                   nrow = 2) {
  if (is.null(admin)) {
    admin <- read_country_boundaries()
  }

  limitVals <- terra::global(
    mySpatRast,
    fun = quantile,
    probs = c(limitValsQuantile, 1 - limitValsQuantile),
    na.rm = TRUE
  )
  limitVals_lower <- as.numeric(limitVals[1])
  limitVals_upper <- as.numeric(limitVals[2])

  mySpatRast <- terra::ifel(
    mySpatRast < limitVals_lower,
    limitVals_lower,
    terra::ifel(mySpatRast > limitVals_upper, limitVals_upper, mySpatRast)
  )

  breaks <- NULL
  if (customBreaks) {
    if (!is.null(customBreaksValues)) {
      breaks <- customBreaksValues
      class_matrix <- cbind(
        breaks[-length(breaks)],
        breaks[-1],
        seq_len(length(breaks) - 1)
      )
      mySpatRast <- terra::classify(mySpatRast, class_matrix, include.lowest = TRUE, right = TRUE)
      mySpatRast <- terra::ifel(mySpatRast == 0, NA, mySpatRast)
      mySpatRast <- as.factor(mySpatRast)
    } else {
      classified <- classify_raster_fisher(mySpatRast, n = customBreaksNumber, samp_prop = samp_prop)
      mySpatRast <- classified$raster
      breaks <- classified$breaks
    }
  }

  g <- ggplot() +
    geom_sf(data = admin, fill = NA, linewidth = 0.1) +
    tidyterra::geom_spatraster(
      data = mySpatRast,
      aes(fill = after_stat(value)),
      maxcell = map_maxcell(mySpatRast)
    ) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.title = element_blank(),
      axis.ticks = element_blank(),
      axis.line = element_blank(),
      panel.border = element_blank(),
      panel.grid = element_blank(),
      legend.position = "bottom",
      legend.key.width = unit(1.2, "cm"),
      legend.key.height = unit(0.3, "cm"),
      legend.title = element_text(margin = margin(b = 12, r = 10)),
      legend.box.margin = margin(-20, 0, 0, 0),
      plot.title = element_text(hjust = 0.5),
      plot.margin = margin(t = 2, r = 5, b = 2, l = 5, unit = "pt")
    )

  if (customBreaks) {
    class_ids <- as.character(seq_len(length(breaks) - 1))
    named_palette <- stats::setNames(color_palette, class_ids)
    labels <- paste0(
      round(breaks[-length(breaks)], label_value_rounding),
      " - ",
      round(breaks[-1], label_value_rounding)
    )
    g <- g +
      scale_fill_manual(
        values = named_palette,
        limits = class_ids,
        breaks = class_ids,
        labels = labels,
        na.value = "transparent",
        drop = FALSE,
        name = name
      ) +
      guides(fill = guide_legend(nrow = nrow, byrow = FALSE, title.position = "top", title.hjust = 0.5))
  } else {
    g <- g + scale_fill_viridis_c(option = "magma", name = legendTitle, na.value = "transparent")
  }

  if (displayProj == "mollweide") {
    g <- g + coord_sf(crs = "+proj=moll +datum=WGS84")
  }

  g
}

plot_country_value_func_local <- function(mySfPolygons,
                                          field,
                                          limitValsQuantile,
                                          displayProj = "planar",
                                          customBreaks = FALSE,
                                          customBreaksNumber = 7,
                                          customBreaksValues = NULL,
                                          color_palette,
                                          name,
                                          label_value_rounding,
                                          nrow = 2) {
  mySfPolygons_df <- mySfPolygons |>
    st_drop_geometry() |>
    as_tibble()

  limitVals <- quantile(mySfPolygons_df[[field]], c(limitValsQuantile, 1 - limitValsQuantile), na.rm = TRUE)
  limitVals_lower <- as.numeric(limitVals[1])
  limitVals_upper <- as.numeric(limitVals[2])

  mySfPolygons <- mySfPolygons |>
    mutate("{field}" := if_else(
      .data[[field]] < limitVals_lower,
      limitVals_lower,
      if_else(.data[[field]] > limitVals_upper, limitVals_upper, .data[[field]])
    ))

  if (customBreaks) {
    values <- mySfPolygons[[field]]
    breaks <- customBreaksValues %||%
      classInt::classIntervals(values, n = customBreaksNumber, style = "fisher", samp_prop = 1)$brks
    mySfPolygons$classified <- cut(values, breaks = breaks, labels = FALSE, include.lowest = TRUE, right = TRUE)
    mySfPolygons$classified <- ifelse(mySfPolygons$classified == 0, NA, mySfPolygons$classified)
    mySfPolygons$classified <- as.factor(mySfPolygons$classified)
    fill_aes <- aes(fill = classified)
  } else {
    breaks <- NULL
    fill_aes <- aes(fill = .data[[field]])
  }

  g <- ggplot(mySfPolygons) +
    geom_sf(fill_aes, linewidth = 0.1) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.title = element_blank(),
      axis.ticks = element_blank(),
      axis.line = element_blank(),
      panel.border = element_blank(),
      panel.grid = element_blank(),
      legend.position = "bottom",
      legend.key.width = unit(1.2, "cm"),
      legend.key.height = unit(0.3, "cm"),
      legend.title = element_text(margin = margin(b = 12, r = 10)),
      legend.box.margin = margin(-20, 0, 0, 0),
      plot.title = element_text(hjust = 0.5),
      plot.margin = margin(t = 2, r = 5, b = 2, l = 5, unit = "pt")
    )

  if (customBreaks) {
    class_ids <- as.character(seq_len(length(breaks) - 1))
    named_palette <- stats::setNames(color_palette, class_ids)
    labels <- paste0(
      round(breaks[-length(breaks)], label_value_rounding),
      " - ",
      round(breaks[-1], label_value_rounding)
    )
    g <- g +
      scale_fill_manual(
        values = named_palette,
        limits = class_ids,
        breaks = class_ids,
        labels = labels,
        na.value = "transparent",
        drop = FALSE,
        name = name
      ) +
      guides(fill = guide_legend(nrow = nrow, byrow = FALSE, title.position = "top", title.hjust = 0.5))
  } else {
    g <- g + scale_fill_viridis_c(option = "magma", name = name, na.value = "transparent")
  }

  if (displayProj == "mollweide") {
    g <- g + coord_sf(crs = "+proj=moll +datum=WGS84")
  }

  g
}

refined_map_theme <- theme(
  legend.position = "bottom",
  plot.margin = margin(t = 2, r = 2, b = 2, l = 2, unit = "pt"),
  legend.title = element_text(size = 8),
  legend.text = element_text(size = 7)
)

quantile_tbl <- function(r, label) {
  terra::global(r, fun = quantile, probs = c(0, 0.025, 0.5, 0.975, 0.99, 0.999, 1), na.rm = TRUE) |>
    as.data.frame() |>
    tibble::rownames_to_column("probability") |>
    mutate(raster = label, .before = 1)
}
