# Supplementary Fig. 15: raster loss maps.
#
# Output basename:
#   SuppFig15_MapPixelLevelLosses

source(file.path("code", "_shared", "loss_figure_helpers.R"))

fig_data <- read_loss_figure_data("SuppFig15Data_MapPixelLevelLosses")
admin <- fig_data$admin
loss_rasters <- lapply(fig_data$loss_rasters, unwrap_figure_raster)

loss_raster_plot <- plot_grid(
  plot_loss_raster(loss_rasters[["production"]], 30, "Reds", expression("Production loss " * (tonnes ~ year^-1)), seq(0, 30, 5), c("0", "5", "10", "15", "20", "25", ">30"), c(0, 30), admin) + refined_theme,
  plot_loss_raster(loss_rasters[["calories"]], 80, "Reds", expression("Calories loss " * (million ~ kcal ~ year^-1)), seq(0, 80, 10), c("0", "10", "20", "30", "40", "50", "60", "70", ">80"), c(0, 80), admin) + refined_theme,
  plot_loss_raster(loss_rasters[["proteins"]], 2, "Reds", expression("Proteins loss " * (tonnes ~ year^-1)), seq(0, 2, 0.25), c("0", "0.25", "0.5", "0.75", "1", "1.25", "1.5", "1.75", ">2"), c(0, 2), admin) + refined_theme,
  plot_loss_raster(loss_rasters[["revenues"]], 7, "Reds", expression("Revenue loss " * (thousand ~ USD ~ year^-1)), seq(0, 7, 1), c("0", "1", "2", "3", "4", "5", "6", ">7"), c(0, 7), admin, transform = function(x) x / 1000) + refined_theme,
  labels = c("A", "B", "C", "D"),
  label_size = 12,
  ncol = 2,
  scale = 1
)

ggsave(path_figure("SuppFig15_MapPixelLevelLosses.png"), loss_raster_plot, width = 18, height = 12, unit = "cm", dpi = 300)
ggsave(path_figure("SuppFig15_MapPixelLevelLosses.pdf"), loss_raster_plot, width = 18, height = 12, unit = "cm")
