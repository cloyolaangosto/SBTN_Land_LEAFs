# Supplementary Fig. 12: CATE map for tree cover debt.
#
# Output basename:
#   SuppFig12_MapCateTreeCover

source(file.path("code", "_shared", "cate_map_helpers.R"))

fig_data <- read_cate_figure_data("SuppFig12Data_MapCateTreeCover")
admin <- fig_data$admin
cate <- unwrap_figure_raster(fig_data$rasters$cate)
p <- plot_individual_cate_raster_by_id("TREECOVER", cate, admin) + refined_theme_sum

ggsave(path_figure("SuppFig12_MapCateTreeCover.png"), p, width = 18, height = 10, unit = "cm", dpi = 300)
ggsave(path_figure("SuppFig12_MapCateTreeCover.pdf"), p, width = 18, height = 10, unit = "cm")
