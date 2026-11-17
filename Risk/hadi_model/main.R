# Main routine replication runner.
#
# Run this script from the root directory of the replication package. It creates
# figure-data files, regenerates figures from those saved figure-data files, and
# regenerates the supplementary tables that are produced by code.

rootdir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)

if (!file.exists(file.path(rootdir, "code", "_shared", "paths.R"))) {
  stop(
    "main.R must be run from the replication package root directory.",
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

run_script(file.path("code", "00_run", "01_create_figure_data.R"))
run_script(file.path("code", "00_run", "02_plot_figures.R"))
run_script(file.path("code", "00_run", "03_create_tables.R"))

message("\nRoutine replication run completed.")

