# Main Fig. 6: country-aggregated relative production losses.
#
# Output basename:
#   Fig6_MapRelativeProductionLosses

source(file.path("code", "_shared", "loss_figure_helpers.R"))

fig_data <- read_loss_figure_data("Fig6Data_MapRelativeProductionLosses")
country_loss_merged_sf <- fig_data$country_loss_sf
bar_data <- fig_data$barplot_panel_B

fig6_plot <- plot_grid(
  plot_countrySum_relative_loss_percent(country_loss_merged_sf) + refined_theme,
  make_top_bar_preselected(bar_data, "Relative loss (% of country production)") + refined_theme_barplot_2,
  labels = c("A", "B"),
  label_size = 12,
  ncol = 2,
  rel_widths = c(1.8, 1),
  scale = 1
)

ggsave(path_figure("Fig6_MapRelativeProductionLosses.png"), fig6_plot, width = 18, height = 6.8, unit = "cm", dpi = 300)
ggsave(path_figure("Fig6_MapRelativeProductionLosses.pdf"), fig6_plot, width = 18, height = 6.8, unit = "cm")
