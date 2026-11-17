# Supplementary Fig. 13: CATE map for erosion with causal forest tuning set to all.
#
# Output basename:
#   SuppFig13_MapCateErosionTuned

source(file.path("code", "_shared", "cate_map_helpers.R"))

fig_data <- read_cate_figure_data("SuppFig13Data_MapCateErosionTuned")
admin <- fig_data$admin
yield_gap <- unwrap_figure_raster(fig_data$yield_gap)
p <- plot_other_cf_experiment(fig_data$data, admin, yield_gap) + refined_theme_sum

ggsave(path_figure("SuppFig13_MapCateErosionTuned.png"), p, width = 18, height = 10, unit = "cm", dpi = 300)
ggsave(path_figure("SuppFig13_MapCateErosionTuned.pdf"), p, width = 18, height = 10, unit = "cm")
