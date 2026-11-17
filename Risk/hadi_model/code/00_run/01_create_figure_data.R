# Create figure-data files used by the manuscript figure plotting scripts.
#
# This routine uses the processed datasets and saved model/CATE/loss outputs
# included in the replication package. It does not refit models or rebuild the
# full analysis-ready dataset from raw raster inputs.

rootdir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
if (!file.exists(file.path(rootdir, "code", "_shared", "paths.R"))) {
  stop("Run this script from the replication package root directory.", call. = FALSE)
}

run_script <- function(path) {
  script <- file.path(rootdir, path)
  if (!file.exists(script)) {
    stop("Missing script: ", path, call. = FALSE)
  }
  message("\n--- Running ", path, " ---")
  source(script, echo = TRUE, chdir = FALSE)
}

scripts <- c(
  file.path("code", "03_figures", "figure_data", "Fig1Data_MapLandDegradationYieldGaps.R"),
  file.path("code", "03_figures", "figure_data", "Fig2Data_CoefPlotModelSpecifications.R"),
  file.path("code", "03_figures", "figure_data", "Fig3Data_CoefPlotYieldGapActualYield.R"),
  file.path("code", "03_figures", "figure_data", "SuppFig4Data_CoefPlotResidualizedTreatmentsRobustness.R"),
  file.path("code", "03_figures", "figure_data", "SuppFig5Data_CoefPlotCoarserResolutionsRobustness.R"),
  file.path("code", "03_figures", "figure_data", "SuppFig6Data_CoefPlotAttainableYield.R"),
  file.path("code", "03_figures", "figure_data", "FigureData_CATE_maps.R"),
  file.path("code", "03_figures", "figure_data", "FigureData_loss_figures.R"),
  file.path("code", "03_figures", "figure_data", "SuppFig18Data_CateByCropType.R"),
  file.path("code", "03_figures", "figure_data", "SuppFig19Data_MapSoilErosionIntensity.R")
)

invisible(lapply(scripts, run_script))
message("\nFigure-data generation completed.")

