# Supplementary Fig. 16: continuous country-sum loss maps.
#
# Output basename:
#   SuppFig16_MapCountryLossesContinuous

source(file.path("code", "_shared", "loss_figure_helpers.R"))

fig_data <- read_loss_figure_data("SuppFig16Data_MapCountryLossesContinuous")
country_abs <- fig_data$country_loss_sf

country_sum_continuous <- plot_grid(
  make_country_map_continuous(country_abs, "countrySum_loss_production_tonnes_thousand", expression("Production loss " * (thousands ~ tonnes ~ year^-1)), 60, seq(0, 60, 10), c("0", "10", "20", "30", "40", "50", ">60"), c(0, 60)) + country_map_theme,
  make_country_map_continuous(country_abs, "countrySum_loss_calories_billion_kcal", expression("Calories loss " * (billions ~ kcal ~ year^-1)), 200, seq(0, 200, 25), c("0", "25", "50", "75", "100", "125", "150", "175", ">200"), c(0, 200)) + country_map_theme,
  make_country_map_continuous(country_abs, "countrySum_loss_proteins_tonnes", expression("Proteins loss " * (tonnes ~ year^-1)), 5000, seq(0, 5000, 1000), c("0", "1000", "2000", "3000", "4000", ">5000"), c(0, 5000)) + country_map_theme,
  make_country_map_continuous(country_abs, "countrySum_loss_revenues_million_USD", expression("Revenue loss " * (millions ~ USD ~ year^-1)), 9, seq(0, 9, 1), c("0", "1", "2", "3", "4", "5", "6", "7", "8", ">9"), c(0, 9)) + country_map_theme,
  labels = c("A", "B", "C", "D"),
  label_size = 12,
  label_x = 0,
  label_y = 1,
  ncol = 2,
  rel_heights = c(1, 1),
  scale = 1
)

ggsave(path_figure("SuppFig16_MapCountryLossesContinuous.png"), country_sum_continuous, width = 18, height = 12, unit = "cm", dpi = 300)
ggsave(path_figure("SuppFig16_MapCountryLossesContinuous.pdf"), country_sum_continuous, width = 18, height = 12, unit = "cm")
