# Supplementary Fig. 10: CATE map for soil organic carbon debt.
#
# Output basename:
#   SuppFig10_MapCateSoc

source(file.path("code", "_shared", "cate_map_helpers.R"))

fig_data <- read_cate_figure_data("SuppFig10Data_MapCateSoc")
admin <- fig_data$admin
cate <- unwrap_figure_raster(fig_data$rasters$cate)
p <- plot_individual_cate_raster_by_id("SOC", cate, admin) + refined_theme_sum

ggsave(path_figure("SuppFig10_MapCateSoc.png"), p, width = 18, height = 10, unit = "cm", dpi = 300)
ggsave(path_figure("SuppFig10_MapCateSoc.pdf"), p, width = 18, height = 10, unit = "cm")
