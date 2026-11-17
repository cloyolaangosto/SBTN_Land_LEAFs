# Supplementary Fig. 14: CATE map for erosion with separate treatment/outcome models.
#
# Output basename:
#   SuppFig14_MapCateErosionSeparateModels

source(file.path("code", "_shared", "cate_map_helpers.R"))

fig_data <- read_cate_figure_data("SuppFig14Data_MapCateErosionSeparateModels")
admin <- fig_data$admin
yield_gap <- unwrap_figure_raster(fig_data$yield_gap)
p <- plot_other_cf_experiment(fig_data$data, admin, yield_gap) + refined_theme_sum

ggsave(path_figure("SuppFig14_MapCateErosionSeparateModels.png"), p, width = 18, height = 10, unit = "cm", dpi = 300)
ggsave(path_figure("SuppFig14_MapCateErosionSeparateModels.pdf"), p, width = 18, height = 10, unit = "cm")
