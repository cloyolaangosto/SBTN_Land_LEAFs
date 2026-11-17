# Generate Supplementary Table 3: country-level absolute loss estimates.
#
# Countries are ranked by production loss. The table reports the top 10
# countries by production loss and includes the corresponding calorie, protein,
# and revenue losses from the same country-level loss object used for Fig. 5.

library(dplyr)
library(readr)
library(sf)

source(file.path("code", "_shared", "loss_figure_helpers.R"))

assert_files_exist(country_loss_path)

table_dir <- file.path(manuscript_output_dir, "tables")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

country_loss <- read_country_loss() |>
  st_drop_geometry()

supp_table3 <- country_loss |>
  filter(!is.na(countrySum_loss_production_tonnes_thousand)) |>
  arrange(desc(countrySum_loss_production_tonnes_thousand)) |>
  slice_head(n = 10) |>
  mutate(
    rank_by_production_loss = row_number(),
    persons_year_at_2000_kcal_per_day = countrySum_loss_calories_billion_kcal * 1e9 / (2000 * 365)
  ) |>
  transmute(
    rank_by_production_loss,
    iso3 = ISO3,
    country = NAME,
    production_loss_thousand_tonnes_per_year = countrySum_loss_production_tonnes_thousand,
    calorie_loss_billion_kcal_per_year = countrySum_loss_calories_billion_kcal,
    persons_year_at_2000_kcal_per_day,
    protein_loss_tonnes_per_year = countrySum_loss_proteins_tonnes,
    revenue_loss_million_usd_per_year = countrySum_loss_revenues_million_USD
  )

write_csv(supp_table3, file.path(table_dir, "SuppTable3_CountryLossesTop10.csv"))
write_csv(supp_table3, path_figure_data("SuppTable3Data_CountryLossesTop10.csv"))
saveRDS(supp_table3, path_figure_data("SuppTable3Data_CountryLossesTop10.rds"))

message("Wrote Supplementary Table 3 outputs to: ", table_dir)
