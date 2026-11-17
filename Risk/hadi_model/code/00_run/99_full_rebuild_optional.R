# Optional full upstream rebuild runner.
#
# The default replication entry point is `main.R`, which uses included saved
# outputs. This script is provided only for users who intentionally want to
# rebuild upstream data/model/loss products. Several steps are computationally
# intensive; causal forest fitting is controlled by a separate environment flag.

rootdir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
if (!file.exists(file.path(rootdir, "code", "_shared", "paths.R"))) {
  stop("Run this script from the replication package root directory.", call. = FALSE)
}

if (!identical(Sys.getenv("LDYG_RUN_FULL_REBUILD"), "true")) {
  stop(
    paste(
      "Full upstream rebuild is disabled by default.",
      "Set LDYG_RUN_FULL_REBUILD=true to run the R rebuild steps intentionally.",
      "The Google Earth Engine export script must be run separately in Python/GEE.",
      sep = "\n"
    ),
    call. = FALSE
  )
}

run_script <- function(path) {
  script <- file.path(rootdir, path)
  if (!file.exists(script)) {
    stop("Missing script: ", path, call. = FALSE)
  }
  message("\n--- Running ", path, " ---")
  source(script, echo = TRUE, chdir = FALSE)
}

message("The Google Earth Engine export is not launched by this R script.")
message("Run code/01_data/01_export_gee_stack.py separately if the raster stack must be rebuilt.")

core_scripts <- c(
  file.path("code", "01_data", "02_build_stage2_grid_table.R"),
  file.path("code", "01_data", "03_build_final_analysis_table.R"),
  file.path("code", "01_data", "04_create_regression_dataset.R"),
  file.path("code", "02_models", "01_fit_regression_models.R"),
  file.path("code", "02_models", "03_create_cate_inspection_outputs.R"),
  file.path("code", "02_models", "04_compute_loss_rasters_and_tables.R")
)

invisible(lapply(core_scripts, run_script))

if (identical(Sys.getenv("LDYG_RUN_CAUSAL_FOREST"), "true")) {
  run_script(file.path("code", "02_models", "02_fit_causal_forests.R"))
} else {
  message("\nSkipping causal forest fitting. Set LDYG_RUN_CAUSAL_FOREST=true to run it intentionally.")
}

run_script(file.path("code", "00_run", "01_create_figure_data.R"))
run_script(file.path("code", "00_run", "02_plot_figures.R"))
run_script(file.path("code", "00_run", "03_create_tables.R"))

message("\nOptional full rebuild runner completed.")

