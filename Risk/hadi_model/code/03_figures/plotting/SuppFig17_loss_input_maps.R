# Supplementary Fig. 17: input maps for production/calorie/protein/revenue calculations.
#
# Output basename:
#   SuppFig17_MapInputDataLossCalculations

source(file.path("code", "_shared", "loss_figure_helpers.R"))

fig_data <- read_loss_figure_data("SuppFig17Data_MapInputDataLossCalculations")
admin <- fig_data$admin
input_rasters <- lapply(fig_data$input_rasters, unwrap_figure_raster)

input_raster_plot <- plot_grid(
  plot_input_raster(input_rasters[["harvest_area"]], 10000, "Reds", expression("Harvested area " * (ha ~ yr^-1)), seq(0, 10000, 2000), c("0", "2000", "4000", "6000", "8000", ">10000"), c(0, 10000), admin, zero_to_na = TRUE) + input_raster_theme,
  plot_input_raster(input_rasters[["attainable_yield"]], 25, "Reds", expression("Attainable yield " * (tonnes ~ ha^-1 ~ yr^-1)), seq(0, 25, 5), c("0", "5", "10", "15", "20", ">25"), c(0, 25), admin) + input_raster_theme,
  plot_input_raster(input_rasters[["price"]], 540, "Reds", expression("Producer price " * (USD ~ tonne^-1)), seq(0, 540, 90), c("0", "90", "180", "270", "360", "450", ">540"), c(0, 541), admin) + input_raster_theme,
  plot_input_raster(input_rasters[["calories"]], 3, "Reds", expression("Calories " * (million ~ kcal ~ tonne^-1)), seq(0, 3, 0.5), c("0", "0.5", "1.0", "1.5", "2.0", "2.5", ">3.0"), c(0, 3), admin, transform = function(x) x / 1e6) + input_raster_theme,
  plot_input_raster(input_rasters[["proteins"]], 0.09, "Reds", expression("Proteins " * (tonnes ~ tonne^-1)), seq(0, 0.09, 0.015), c("0", "0.015", "0.03", "0.045", "0.06", "0.075", ">0.09"), c(0, 0.09), admin) + input_raster_theme,
  labels = c("A", "B", "C", "D", "E"),
  label_size = 12,
  ncol = 2,
  scale = 1
)

ggsave(path_figure("SuppFig17_MapInputDataLossCalculations.png"), input_raster_plot, width = 18, height = 16, unit = "cm", dpi = 300)
ggsave(path_figure("SuppFig17_MapInputDataLossCalculations.pdf"), input_raster_plot, width = 18, height = 16, unit = "cm")
