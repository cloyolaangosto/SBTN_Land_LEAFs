# Generate Main Fig. 1 and Supplementary Fig. 1.
#
# The maps place the five land-degradation debt variables and 2010 crop yield
# gaps on the common 10 km analysis grid. Main Fig. 1 uses continuous colour
# scales; Supplementary Fig. 1 uses Fisher classes for the same variables.

library(cowplot)
library(colorspace)
library(dplyr)
library(ggplot2)
library(readr)
library(terra)

source(file.path("code", "_shared", "paths.R"))
source(file.path("code", "_shared", "map_helpers.R"))

fig1_figure_data <- readRDS(path_figure_data("Fig1Data_MapLandDegradationYieldGaps.rds"))
fig1_stack <- terra::unwrap(fig1_figure_data$rasters)

soilErosionDebt_rst <- fig1_stack[["soilErosionDebt"]]
soilCompactionDebt_rst <- fig1_stack[["soilCompactionDebt"]]
soilOrganicCarbonDebt_rst <- fig1_stack[["soilOrganicCarbonDebt"]]
soilWaterDebt_rst <- fig1_stack[["soilWaterDebt"]]
treeCoverDebt_rst <- fig1_stack[["treeCoverDebt"]]
yieldGapPercent_rst <- fig1_stack[["AllCrops2010YieldGapPercent"]]

color_palette_7 <- c("#FFFFB2", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026")

plot_soilErosionDebt_fisher_fin <- plot_raster_func_local(
  soilErosionDebt_rst, 0, "%", "mollweide", admin = NULL, customBreaks = TRUE, customBreaksNumber = 7, customBreaksValues = NULL, samp_prop = 1,
  color_palette = color_palette_7, name = expression("Soil erosion debt " * (Mg ~ ha^-1 ~ yr^-1)), label_value_rounding = 0, nrow = 2
)
plot_soilCompactionDebt_fisher_fin <- plot_raster_func_local(
  soilCompactionDebt_rst, 0, "%", "mollweide", admin = NULL, customBreaks = TRUE, customBreaksNumber = 7, customBreaksValues = NULL, samp_prop = 1,
  color_palette = color_palette_7, name = expression("Soil compaction debt (unitless)"), label_value_rounding = 1, nrow = 2
)
plot_soilOrganicCarbonDebt_fisher_fin <- plot_raster_func_local(
  soilOrganicCarbonDebt_rst, 0, "%", "mollweide", admin = NULL, customBreaks = TRUE, customBreaksNumber = 7, customBreaksValues = NULL, samp_prop = 1,
  color_palette = color_palette_7, name = expression("Soil organic carbon debt " * (Mg ~ ha^-1)), label_value_rounding = 1, nrow = 2
)
plot_soilWaterDebt_fisher_fin <- plot_raster_func_local(
  soilWaterDebt_rst, 0, "%", "mollweide", admin = NULL, customBreaks = TRUE, customBreaksNumber = 7, customBreaksValues = NULL, samp_prop = 1,
  color_palette = color_palette_7, name = expression("Soil water debt " * (m^3 ~ m^-3)), label_value_rounding = 3, nrow = 2
)
plot_treeCoverDebt_fisher_fin <- plot_raster_func_local(
  treeCoverDebt_rst, 0, "%", "mollweide", admin = NULL, customBreaks = TRUE, customBreaksNumber = 7, customBreaksValues = NULL, samp_prop = 1,
  color_palette = color_palette_7, name = "Tree cover debt (ha)", label_value_rounding = 0, nrow = 2
)
plot_yieldGapPercent_fisher_fin <- plot_raster_func_local(
  yieldGapPercent_rst, 0, "%", "mollweide", admin = NULL, customBreaks = TRUE, customBreaksNumber = 7, customBreaksValues = NULL, samp_prop = 1,
  color_palette = color_palette_7, name = "Crop yield gaps (%)", label_value_rounding = 0, nrow = 2
)

refined_theme <- theme(
  legend.key.width = unit(0.28, "cm"),
  legend.key.height = unit(0.08, "cm"),
  legend.title = element_text(margin = margin(b = 1, t = 1), size = 7),
  legend.text = element_text(size = 6),
  legend.box.margin = margin(t = -24, b = 0),
  legend.margin = margin(t = 0, r = 0, b = 0, l = 0),
  legend.spacing.y = unit(0, "pt"),
  plot.margin = margin(t = -10, r = -10, b = 12, l = -10, unit = "pt"),
  plot.title = element_text(hjust = 0.5, size = 10)
)

viridis_theme <- theme(
  legend.key.width = unit(0.1, "cm"),
  legend.key.height = unit(0.2, "cm"),
  legend.title = element_text(margin = margin(b = 2, t = 2), size = 8),
  legend.text = element_text(size = 7),
  legend.box.margin = margin(t = -35),
  plot.margin = margin(t = -10, r = -10, b = -10, l = -10, unit = "pt"),
  plot.title = element_text(hjust = 0.5, size = 10)
)

make_continuous_plot <- function(r, clamp, palette, name, breaks, labels, limits = NULL, quantile = 0, rev = FALSE, viridis_direction = NULL) {
  g <- plot_raster_func_local(
    custom_clamp_func(r, clamp), quantile, "%", "mollweide", admin = NULL, customBreaks = FALSE
  )
  if (!is.null(viridis_direction)) {
    return(g + ggplot2::scale_fill_viridis_c(
      option = "viridis",
      direction = viridis_direction,
      name = name,
      breaks = breaks,
      labels = labels,
      limits = limits,
      na.value = "transparent",
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(5, "cm"),
        barheight = unit(0.2, "cm")
      )
    ))
  }
  g +
    colorspace::scale_fill_continuous_sequential(
      palette = palette,
      rev = rev,
      name = name,
      breaks = breaks,
      labels = labels,
      limits = limits,
      na.value = "transparent",
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(5, "cm"),
        barheight = unit(0.2, "cm")
      )
    )
}

combined_plot_viridis <- plot_grid(
  make_continuous_plot(soilErosionDebt_rst, 25, "Reds2", expression("Soil erosion debt " * (Mg ~ ha^-1 ~ yr^-1)), seq(0, 25, 5), c("0", "5", "10", "15", "20", ">25"), rev = TRUE) + viridis_theme,
  make_continuous_plot(soilCompactionDebt_rst, 2, "Reds", expression("Soil compaction debt (unitless)"), seq(0, 2, 0.5), c("0", "0.5", "1", "1.5", ">2"), limits = c(0, 2), rev = TRUE) + viridis_theme,
  make_continuous_plot(soilOrganicCarbonDebt_rst, 4, "Peach", expression("Soil organic carbon debt " * (Mg ~ ha^-1)), seq(0, 4, 1), c("0", "1", "2", "3", ">4"), rev = TRUE) + viridis_theme,
  make_continuous_plot(soilWaterDebt_rst, 0.025, "Blues", expression("Soil water debt " * (m^3 ~ m^-3)), seq(0, 0.025, 0.005), c("0", "0.005", "0.01", "0.015", "0.02", ">0.025"), rev = TRUE) + viridis_theme,
  make_continuous_plot(treeCoverDebt_rst, 2500, "Greens", "Tree cover debt (ha)", seq(0, 2500, 500), c("0", "500", "1000", "1500", "2000", ">2500"), limits = c(0, 2500), rev = TRUE) + viridis_theme,
  make_continuous_plot(yieldGapPercent_rst, 100, "Viridis", "Crop yield gaps (%)", seq(0, 100, 20), c("0", "20", "40", "60", "80", "100"), limits = c(0, 100), viridis_direction = -1) + viridis_theme,
  labels = c("A", "B", "C", "D", "E", "F"),
  label_size = 12,
  label_x = 0,
  label_y = 1,
  ncol = 2,
  rel_heights = c(1, 1),
  scale = 1
)

combined_plot <- plot_grid(
  plot_soilErosionDebt_fisher_fin + refined_theme,
  plot_soilCompactionDebt_fisher_fin + refined_theme,
  plot_soilOrganicCarbonDebt_fisher_fin + refined_theme,
  plot_soilWaterDebt_fisher_fin + refined_theme,
  plot_treeCoverDebt_fisher_fin + refined_theme,
  plot_yieldGapPercent_fisher_fin + refined_theme,
  labels = c("A", "B", "C", "D", "E", "F"),
  label_size = 12,
  label_x = 0,
  label_y = 1,
  ncol = 2,
  rel_heights = c(1, 1),
  scale = 1
)

ggsave(path_figure("Fig1_MapLandDegradationYieldGaps.png"), combined_plot_viridis, width = 18, height = 16, unit = "cm", dpi = 300)
ggsave(path_figure("Fig1_MapLandDegradationYieldGaps.pdf"), combined_plot_viridis, width = 18, height = 16, unit = "cm")
ggsave(path_figure("SuppFig1_MapLandDegradationYieldGapFisher7.png"), combined_plot, width = 18, height = 16, unit = "cm", dpi = 300)
ggsave(path_figure("SuppFig1_MapLandDegradationYieldGapFisher7.pdf"), combined_plot, width = 18, height = 16, unit = "cm")
