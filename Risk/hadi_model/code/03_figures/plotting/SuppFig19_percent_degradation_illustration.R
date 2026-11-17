# Generate Supplementary Fig. 19: soil-erosion intensity illustration.
#
# This script generates the base figure used for the manually edited
# supplementary figure.

library(terra)
library(tmap)

source(file.path("code", "_shared", "paths.R"))

tmap_mode("plot")

fig_data <- readRDS(path_figure_data("SuppFig19Data_MapSoilErosionIntensity.rds"))
class_stack <- terra::unwrap(fig_data$class_rasters)

data("World", package = "tmap")
World_no_antarctica <- World[World$continent != "Antarctica", ]

plot_class_map <- function(r_class, title) {
  tm_shape(r_class) +
    tm_raster(
      style = "cat",
      palette = c("#4575B4", "#D73027"),
      title = "Intensity",
      legend.show = TRUE
    ) +
    tm_shape(World_no_antarctica) +
    tm_borders(col = "gray20", lwd = 0.5) +
    tm_layout(
      main.title = title,
      main.title.size = 0.8,
      legend.outside = FALSE,
      legend.position = c("left", "bottom"),
      legend.bg.color = "white",
      legend.bg.alpha = 0.9,
      legend.frame = TRUE,
      frame = FALSE
    )
}

p1 <- plot_class_map(class_stack[["double"]], "Typical vs Double")
p2 <- plot_class_map(class_stack[["triple"]], "Typical vs Triple")
p3 <- plot_class_map(class_stack[["high_4x"]], "Typical vs High (4x)")
p4 <- plot_class_map(class_stack[["extreme"]], "Typical vs Extreme")

combined_map <- tmap_arrange(p1, p2, p3, p4, ncol = 2)

tmap_save(
  combined_map,
  filename = path_figure("SuppFig19_MapSoilErosionIntensity.png"),
  width = 8.3,
  height = 11.7,
  units = "in",
  dpi = 300
)
tmap_save(
  combined_map,
  filename = path_figure("SuppFig19_MapSoilErosionIntensity.pdf"),
  width = 8.3,
  height = 11.7,
  units = "in"
)
