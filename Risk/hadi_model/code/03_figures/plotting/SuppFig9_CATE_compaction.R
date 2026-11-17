# Supplementary Fig. 9: CATE map for soil compaction debt.
#
# Output basename:
#   SuppFig9_MapCateCompaction

source(file.path("code", "_shared", "cate_map_helpers.R"))

fig_data <- read_cate_figure_data("SuppFig9Data_MapCateCompaction")
admin <- fig_data$admin
cate <- unwrap_figure_raster(fig_data$rasters$cate)
p <- plot_individual_cate_raster_by_id("COMPACTION", cate, admin) + refined_theme_sum

ggsave(path_figure("SuppFig9_MapCateCompaction.png"), p, width = 18, height = 10, unit = "cm", dpi = 300)
ggsave(path_figure("SuppFig9_MapCateCompaction.pdf"), p, width = 18, height = 10, unit = "cm")
