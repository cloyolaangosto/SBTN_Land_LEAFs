# Supplementary Fig. 8: CATE map for soil erosion debt.
#
# Output basename:
#   SuppFig8_MapCateErosion

source(file.path("code", "_shared", "cate_map_helpers.R"))

fig_data <- read_cate_figure_data("SuppFig8Data_MapCateErosion")
admin <- fig_data$admin
cate <- unwrap_figure_raster(fig_data$rasters$cate)
p <- plot_individual_cate_raster_by_id("EROSION", cate, admin) + refined_theme_sum

ggsave(path_figure("SuppFig8_MapCateErosion.png"), p, width = 18, height = 10, unit = "cm", dpi = 300)
ggsave(path_figure("SuppFig8_MapCateErosion.pdf"), p, width = 18, height = 10, unit = "cm")
