# Generate Supplementary Table 5: summary statistics for the main estimation sample.
#
# The table selects variables used in the reported models, drops incomplete
# observations consistently with the estimation sample, and exports numeric
# summary statistics as CSV.

library(dplyr)
library(readr)
library(stringr)
library(tidyr)

source(file.path("code", "_shared", "paths.R"))

dataset_path <- path_processed("final_dataset", "analysis_ready_data_full.rds")
model_spec_path <- path_processed(
  "models",
  "main_10km",
  "model_spec_variables.rds"
)

assert_files_exist(c(dataset_path, model_spec_path))

df_ok <- readRDS(dataset_path)
reg_varNames <- readRDS(model_spec_path)

reg_varNames_flat <- unlist(reg_varNames) |>
  unname() |>
  unique()

reg_varNames_flat_notLog_notScaled <- reg_varNames_flat |>
  str_replace_all("__rescaled__log", "") |>
  str_replace_all("__log", "") |>
  unique()

preferred_vars <- reg_varNames_flat_notLog_notScaled
fallback_vars <- reg_varNames_flat

selected_vars <- preferred_vars[preferred_vars %in% names(df_ok)]
fallback_selected <- fallback_vars[fallback_vars %in% names(df_ok) & !(fallback_vars %in% selected_vars)]
selected_vars <- unique(c(selected_vars, fallback_selected))

df_ok_tableS1 <- df_ok |>
  select(all_of(selected_vars)) |>
  drop_na()

numeric_summary <- df_ok_tableS1 |>
  select(where(is.numeric)) |>
  summarise(across(
    everything(),
    list(
      N = ~ sum(!is.na(.x)),
      Mean = ~ mean(.x, na.rm = TRUE),
      Median = ~ median(.x, na.rm = TRUE),
      SD = ~ sd(.x, na.rm = TRUE),
      Min = ~ min(.x, na.rm = TRUE),
      Max = ~ max(.x, na.rm = TRUE),
      Quantile_001 = ~ quantile(.x, 0.01, na.rm = TRUE),
      Quantile_0025 = ~ quantile(.x, 0.025, na.rm = TRUE),
      Quantile_0975 = ~ quantile(.x, 0.975, na.rm = TRUE),
      Quantile_099 = ~ quantile(.x, 0.99, na.rm = TRUE)
    ),
    .names = "{.col}__{.fn}"
  )) |>
  tidyr::pivot_longer(everything(), names_to = "variable_stat", values_to = "value") |>
  tidyr::separate_wider_delim(variable_stat, delim = "__", names = c("variable", "stat"), too_many = "merge") |>
  tidyr::pivot_wider(names_from = stat, values_from = value)

table_dir <- file.path(manuscript_output_dir, "tables")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

write_csv(numeric_summary, file.path(table_dir, "SuppTable5_SummaryStatisticsTable.csv"))

message("Wrote Supplementary Table 5 outputs to: ", table_dir)
