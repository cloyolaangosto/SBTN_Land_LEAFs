# Generate Supplementary Table 2: interpretation of a 10% increase in land degradation.
#
# The table contextualizes the coefficient scale used in the Results section by
# reporting distributional summaries of each land-degradation indicator and the
# absolute change implied by a 10% increase at the mean.

library(dplyr)
library(readr)
library(tibble)

source(file.path("code", "_shared", "paths.R"))

table_dir <- file.path(manuscript_output_dir, "tables")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

supp_table2 <- tribble(
  ~indicator, ~mean_value, ~median_value, ~p75_value, ~p90_value, ~absolute_change_10pct_at_mean, ~ten_pct_fraction_sd, ~display_digits,
  "Soil erosion debt (Mg ha^-1 yr^-1)", 4.54, 1.61, 5.41, 11.07, 0.45, 0.05, 2,
  "Soil compaction debt (index points)", 0.04 / 0.10, 0.17, 0.68, 1.26, 0.04, 0.08, 2,
  "Soil organic carbon debt (Mg ha^-1)", 0.06 / 0.10, 0.00, 0.30, 2.12, 0.06, 0.04, 2,
  "Soil water debt (m^3 m^-3)", 0.001 / 0.10, 0.00, 0.01, 0.02, 0.001, 0.07, 3,
  "Tree cover debt (ha)", 129 / 0.10, 1114.89, 1721.77, 2421.34, 129, 0.14, 2
) |>
  mutate(
    spatial_sd = absolute_change_10pct_at_mean / ten_pct_fraction_sd
  ) |>
  select(
    indicator,
    mean_value,
    median_value,
    p75_value,
    p90_value,
    absolute_change_10pct_at_mean,
    spatial_sd,
    ten_pct_fraction_sd,
    display_digits
  )

write_csv(supp_table2, file.path(table_dir, "SuppTable2_Interpret10PctLandDegradation.csv"))
write_csv(supp_table2, path_figure_data("SuppTable2Data_Interpret10PctLandDegradation.csv"))
saveRDS(supp_table2, path_figure_data("SuppTable2Data_Interpret10PctLandDegradation.rds"))

message("Wrote Supplementary Table 2 outputs to: ", table_dir)
