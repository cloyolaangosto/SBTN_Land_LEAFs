# Main Fig. 4: estimated increase in crop yield gaps associated with overall land degradation.
#
# Output basename:
#   Fig4_MapOverallLandDegradationCate

source(file.path("code", "_shared", "cate_map_helpers.R"))

fig_data <- read_cate_figure_data("Fig4Data_MapOverallLandDegradationCate")
admin <- fig_data$admin
cate_sum <- unwrap_figure_raster(fig_data$rasters$cate_sum)
yield_gap <- unwrap_figure_raster(fig_data$rasters$yield_gap)

combined_plot_fig3 <- plot_grid(
  plot_cate_relative_change(cate_sum, admin) + refined_theme_sum,
  plot_cate_absolute_pp_change(cate_sum, yield_gap, admin) + refined_theme_sum,
  labels = c("A", "B"),
  label_size = 12,
  label_x = 0,
  label_y = 1,
  ncol = 1,
  rel_heights = c(1, 1),
  scale = 1
)

ggsave(path_figure("Fig4_MapOverallLandDegradationCate.png"), combined_plot_fig3, width = 18, height = 20, unit = "cm", dpi = 300)
ggsave(path_figure("Fig4_MapOverallLandDegradationCate.pdf"), combined_plot_fig3, width = 18, height = 20, unit = "cm")
