# Supplementary Fig. 7: Fisher-class overall CATE map.
#
# Output basename:
#   SuppFig7_MapOverallLandDegradationCateFisher7

source(file.path("code", "_shared", "cate_map_helpers.R"))

fig_data <- read_cate_figure_data("SuppFig7Data_MapOverallLandDegradationCateFisher7")
admin <- fig_data$admin
cate_sum <- unwrap_figure_raster(fig_data$rasters$cate_sum)

p <- plot_cate_sum_fisher(cate_sum, admin) + refined_theme_fisher

ggsave(path_figure("SuppFig7_MapOverallLandDegradationCateFisher7.png"), p, width = 18, height = 10, unit = "cm", dpi = 300)
ggsave(path_figure("SuppFig7_MapOverallLandDegradationCateFisher7.pdf"), p, width = 18, height = 10, unit = "cm")
