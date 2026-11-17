# Plot manuscript figures from saved figure-data files.
#
# Run `code/00_run/01_create_figure_data.R` first if the figure-data files in
# `output/figure_data/` need to be refreshed.

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
  file.path("code", "03_figures", "plotting", "Fig1_land_degradation_yield_gaps.R"),
  file.path("code", "03_figures", "plotting", "Fig2_regression_specifications.R"),
  file.path("code", "03_figures", "plotting", "Fig3_yield_gap_actual_yield.R"),
  file.path("code", "03_figures", "plotting", "Fig4_overall_land_degradation_effect.R"),
  file.path("code", "03_figures", "plotting", "Fig5_country_loss_maps.R"),
  file.path("code", "03_figures", "plotting", "Fig5_country_loss_barplots.R"),
  file.path("code", "03_figures", "plotting", "Fig5_country_losses_combined.R"),
  file.path("code", "03_figures", "plotting", "Fig6_relative_losses.R"),
  file.path("code", "03_figures", "plotting", "SuppFig4_residualized_treatments.R"),
  file.path("code", "03_figures", "plotting", "SuppFig5_coarser_resolutions.R"),
  file.path("code", "03_figures", "plotting", "SuppFig6_attainable_yield.R"),
  file.path("code", "03_figures", "plotting", "SuppFig7_overall_effect_fisher_map.R"),
  file.path("code", "03_figures", "plotting", "SuppFig8_CATE_erosion.R"),
  file.path("code", "03_figures", "plotting", "SuppFig9_CATE_compaction.R"),
  file.path("code", "03_figures", "plotting", "SuppFig10_CATE_soc.R"),
  file.path("code", "03_figures", "plotting", "SuppFig11_CATE_soilwater.R"),
  file.path("code", "03_figures", "plotting", "SuppFig12_CATE_treecover.R"),
  file.path("code", "03_figures", "plotting", "SuppFig13_CATE_erosion_tuned.R"),
  file.path("code", "03_figures", "plotting", "SuppFig14_CATE_erosion_separate_models.R"),
  file.path("code", "03_figures", "plotting", "SuppFig15_loss_raster_maps.R"),
  file.path("code", "03_figures", "plotting", "SuppFig16_country_loss_maps_continuous.R"),
  file.path("code", "03_figures", "plotting", "SuppFig17_loss_input_maps.R"),
  file.path("code", "03_figures", "plotting", "SuppFig18_crop_type_cate.R"),
  file.path("code", "03_figures", "plotting", "SuppFig19_percent_degradation_illustration.R")
)

invisible(lapply(scripts, run_script))
message("\nFigure plotting completed.")

