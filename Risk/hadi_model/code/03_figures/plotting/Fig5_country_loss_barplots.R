# Generate Main Fig. 5 barplot panels.
#
# The panels rank the countries with the largest estimated production, calorie,
# protein, and revenue losses associated with overall land degradation.
#
# Output basename:
#   Fig5_BarplotCountryLosses

source(file.path("code", "_shared", "loss_figure_helpers.R"))

fig_data <- read_loss_figure_data("Fig5Data_BarplotCountryLosses")
bar_data <- fig_data$barplot_panels_E_G

fig5_plot <- plot_grid(
  make_top_bar_preselected(filter(bar_data, panel == "E"), expression("Production (`000 " * tonnes ~ year^-1 * ")")) + bar_theme,
  make_top_bar_preselected(filter(bar_data, panel == "F"), expression("Calories " * (billions ~ kcal ~ year^-1))) + bar_theme,
  make_top_bar_preselected(filter(bar_data, panel == "G"), expression("Revenue " * (millions ~ USD ~ year^-1))) + bar_theme,
  labels = c("E", "F", "G"),
  label_size = 12,
  label_x = 0,
  label_y = 1,
  ncol = 3,
  rel_heights = c(1, 1),
  scale = 1
)

ggsave(path_figure("Fig5_BarplotCountryLosses.png"), fig5_plot, width = 18, height = 6, unit = "cm", dpi = 300)
ggsave(path_figure("Fig5_BarplotCountryLosses.pdf"), fig5_plot, width = 18, height = 6, unit = "cm")
