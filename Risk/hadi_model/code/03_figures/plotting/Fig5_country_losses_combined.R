# Generate Main Fig. 5 as a combined country-loss figure.
#
# Map panels A-D show country-aggregated losses; barplot panels E-G summarize
# the countries with the largest losses in selected units.
#
# Output basename:
#   Fig5_CountryLosses

source(file.path("code", "_shared", "loss_figure_helpers.R"))

fig_data <- read_loss_figure_data("Fig5Data_CountryLosses")
country_abs <- fig_data$country_loss_sf
bar_data <- fig_data$barplot_panels_E_G

fig5_maps <- plot_grid(
  make_country_map(country_abs, "countrySum_loss_production_tonnes_thousand", expression("Production loss " * (thousands ~ tonnes ~ year^-1)), fisher = TRUE, rounding = 0) + country_map_theme,
  make_country_map(country_abs, "countrySum_loss_calories_billion_kcal", expression("Calories loss " * (billions ~ kcal ~ year^-1)), fisher = TRUE, rounding = 0) + country_map_theme,
  make_country_map(country_abs, "countrySum_loss_proteins_tonnes", expression("Proteins loss " * (tonnes ~ year^-1)), fisher = TRUE, rounding = 0) + country_map_theme,
  make_country_map(country_abs, "countrySum_loss_revenues_million_USD", expression("Revenue loss " * (millions ~ USD ~ year^-1)), fisher = TRUE, rounding = 0) + country_map_theme,
  labels = c("A", "B", "C", "D"),
  label_size = 12,
  label_x = 0,
  label_y = 1,
  ncol = 2,
  rel_heights = c(1, 1),
  scale = 1
)

fig5_barplots <- plot_grid(
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

fig5_combined <- plot_grid(
  fig5_maps,
  fig5_barplots,
  ncol = 1,
  rel_heights = c(2, 1),
  scale = 1
)

ggsave(path_figure("Fig5_CountryLosses.png"), fig5_combined, width = 18, height = 18, unit = "cm", dpi = 300)
ggsave(path_figure("Fig5_CountryLosses.pdf"), fig5_combined, width = 18, height = 18, unit = "cm")
