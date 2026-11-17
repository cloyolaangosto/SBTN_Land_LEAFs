# Generate Main Fig. 5 country-sum Fisher map panels.
#
# The maps show country-aggregated production, calorie, protein, and revenue
# losses associated with overall land degradation.
#
# Output basename:
#   Fig5_MapCountryLosses

source(file.path("code", "_shared", "loss_figure_helpers.R"))

fig_data <- read_loss_figure_data("Fig5Data_MapCountryLosses")
country_abs <- fig_data$country_loss_sf

country_sum_fisher <- plot_grid(
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

ggsave(path_figure("Fig5_MapCountryLosses.png"), country_sum_fisher, width = 18, height = 12, unit = "cm", dpi = 300)
ggsave(path_figure("Fig5_MapCountryLosses.pdf"), country_sum_fisher, width = 18, height = 12, unit = "cm")
