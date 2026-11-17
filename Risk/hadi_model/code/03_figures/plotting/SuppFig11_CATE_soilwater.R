# Supplementary Fig. 11: CATE map for soil water debt.
#
# Output basename:
#   SuppFig11_MapCateSoilWater

source(file.path("code", "_shared", "cate_map_helpers.R"))

fig_data <- read_cate_figure_data("SuppFig11Data_MapCateSoilWater")
admin <- fig_data$admin
cate <- unwrap_figure_raster(fig_data$rasters$cate)
p <- plot_individual_cate_raster_by_id("SOILWATER", cate, admin) + refined_theme_sum

ggsave(path_figure("SuppFig11_MapCateSoilWater.png"), p, width = 18, height = 10, unit = "cm", dpi = 300)
ggsave(path_figure("SuppFig11_MapCateSoilWater.pdf"), p, width = 18, height = 10, unit = "cm")
