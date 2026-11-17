# Create the final variables-only analysis-ready dataset.
#
# This selection/validation stage keeps the variables required by the reported
# regression models and causal-forest CATE inspection. 


library(dplyr)
library(readr)

source(file.path("code", "_shared", "analysis_paths.R"))
source(file.path("code", "01_data", "dataset_contract.R"))

input_dataset <- historical_final_dataset_path()
output_dataset <- analysis_ready_dataset_path()

assert_files_exist(input_dataset)

df <- read_rds(input_dataset)
validation <- validate_analysis_ready_columns(df)


if (!validation$ok) {
  missing <- unique(c(validation$missing_keep, validation$missing_regression, validation$missing_causal_forest))
  stop("Input dataset is missing required analysis-ready columns: ", paste(missing, collapse = ", "), call. = FALSE)
}

df_analysis_ready <- df |>
  select(all_of(analysis_ready_keep_vars))

dir.create(dirname(output_dataset), recursive = TRUE, showWarnings = FALSE)
write_rds(df_analysis_ready, output_dataset)


message("Created variables-only analysis-ready dataset: ", output_dataset)
message("Rows: ", nrow(df_analysis_ready), "; columns: ", ncol(df_analysis_ready), " (from ", ncol(df), ")")
