# Create figure data for Main Fig. 2, Supplementary Fig. 2, and Supplementary Fig. 3.
#
# The figure compares the estimated association between each land-degradation
# debt and crop yield gaps across the reported model specifications. It reads
# saved model and average-slope objects; average-slope computation is handled in
# the model script because it is long-running.

library(broom)
library(cowplot)
library(dplyr)
library(fixest)
library(ggplot2)
library(readr)
library(tibble)

source(file.path("code", "_shared", "paths.R"))

model_dir <- path_processed("models", "main_10km")
model_dir_hc1 <- path_processed("models", "hc1")
model_dir_conley100 <- path_processed("models", "conley_100km")

read_model <- function(...) readRDS(file.path(...))

required_files <- c(
  file.path(model_dir, "model_spec_variables.rds"),
  file.path(model_dir, "model_no_controls.rds"),
  file.path(model_dir, "model_natural_controls.rds"),
  file.path(model_dir, "model_natural_adaptation_controls.rds"),
  file.path(model_dir, "model_full_controls.rds"),
  file.path(model_dir, "model_long_lat_polynomial.rds"),
  file.path(model_dir, "model_crop_type_controls.rds"),
  file.path(model_dir, "model_without_soilgrid_controls.rds"),
  file.path(model_dir, "model_winsorized.rds"),
  file.path(model_dir, "model_field_size_control.rds"),
  file.path(model_dir, "model_soil_depth_control.rds"),
  file.path(model_dir, "model_soil_n_control.rds"),
  file.path(model_dir, "model_water_table_depth_control.rds"),
  file.path(model_dir, "model_road_density_control.rds"),
  file.path(model_dir, "model_travel_time_to_cities_control.rds"),
  file.path(model_dir, "model_without_machinery_control.rds"),
  file.path(model_dir, "model_agricultural_expenditure_control.rds"),
  file.path(model_dir, "model_economic_freedom_control.rds"),
  file.path(model_dir, "model_quadratic_land_degradation_average_slope.rds"),
  file.path(model_dir, "model_interaction_land_degradation_average_slope.rds"),
  file.path(model_dir, "model_interaction_land_degradation_adaptation_average_slope.rds"),
  file.path(model_dir_hc1, "model_baseline.rds"),
  file.path(model_dir_conley100, "model_baseline.rds"),
  file.path(model_dir_conley100, "model_spec_variables.rds"),
  file.path(model_dir_conley100, "model_no_controls.rds"),
  file.path(model_dir_conley100, "model_natural_controls.rds"),
  file.path(model_dir_conley100, "model_natural_adaptation_controls.rds"),
  file.path(model_dir_conley100, "model_full_controls.rds"),
  file.path(model_dir_conley100, "model_long_lat_polynomial.rds"),
  file.path(model_dir_conley100, "model_crop_type_controls.rds"),
  file.path(model_dir_conley100, "model_without_soilgrid_controls.rds"),
  file.path(model_dir_conley100, "model_winsorized.rds"),
  file.path(model_dir_conley100, "model_quadratic_land_degradation_average_slope.rds"),
  file.path(model_dir_conley100, "model_interaction_land_degradation_average_slope.rds"),
  file.path(model_dir_conley100, "model_interaction_land_degradation_adaptation_average_slope.rds")
)
assert_files_exist(required_files)

reg_varNames <- read_model(model_dir, "model_spec_variables.rds")
TREATMENTS <- reg_varNames$TREATMENTS

mf_noControls <- read_model(model_dir, "model_no_controls.rds")
mf_naturalControls <- read_model(model_dir, "model_natural_controls.rds")
mf_naturalControls_adaptControls <- read_model(model_dir, "model_natural_adaptation_controls.rds")
mf_naturalControls_adaptControls_socioeconControls <- read_model(model_dir, "model_full_controls.rds")
mf_addLongLat <- read_model(model_dir, "model_long_lat_polynomial.rds")
mf_addCropType <- read_model(model_dir, "model_crop_type_controls.rds")
mf_rmSoilGrid <- read_model(model_dir, "model_without_soilgrid_controls.rds")
mf_winsor <- read_model(model_dir, "model_winsorized.rds")
mf_addFieldSize <- read_model(model_dir, "model_field_size_control.rds")
mf_addSoilDepth <- read_model(model_dir, "model_soil_depth_control.rds")
mf_addSoilN <- read_model(model_dir, "model_soil_n_control.rds")
mf_addWTD <- read_model(model_dir, "model_water_table_depth_control.rds")
mf_addRoadDensity <- read_model(model_dir, "model_road_density_control.rds")
mf_addTravelCities <- read_model(model_dir, "model_travel_time_to_cities_control.rds")
mf_rmMachine <- read_model(model_dir, "model_without_machinery_control.rds")
mf_addAgriExpend <- read_model(model_dir, "model_agricultural_expenditure_control.rds")
mf_addEconFree <- read_model(model_dir, "model_economic_freedom_control.rds")
mf_addQuadraticLDs_avg_slope <- read_model(model_dir, "model_quadratic_land_degradation_average_slope.rds")
mf_addInteractionLDs_avg_slope <- read_model(model_dir, "model_interaction_land_degradation_average_slope.rds")
mf_addInteractionLDs_addInteractionLDsAdapt_avg_slope <- read_model(model_dir, "model_interaction_land_degradation_adaptation_average_slope.rds")
mf_HC1 <- read_model(model_dir_hc1, "model_baseline.rds")
mf_Conley100 <- read_model(model_dir_conley100, "model_baseline.rds")

mf_conley_noControls <- read_model(model_dir_conley100, "model_no_controls.rds")
mf_conley_naturalControls <- read_model(model_dir_conley100, "model_natural_controls.rds")
mf_conley_naturalControls_adaptControls <- read_model(model_dir_conley100, "model_natural_adaptation_controls.rds")
mf_conley_naturalControls_adaptControls_socioeconControls <- read_model(model_dir_conley100, "model_full_controls.rds")
mf_conley_addLongLat <- read_model(model_dir_conley100, "model_long_lat_polynomial.rds")
mf_conley_addCropType <- read_model(model_dir_conley100, "model_crop_type_controls.rds")
mf_conley_rmSoilGrid <- read_model(model_dir_conley100, "model_without_soilgrid_controls.rds")
mf_conley_winsor <- read_model(model_dir_conley100, "model_winsorized.rds")
mf_conley_addQuadraticLDs_avg_slope <- read_model(model_dir_conley100, "model_quadratic_land_degradation_average_slope.rds")
mf_conley_addInteractionLDs_avg_slope <- read_model(model_dir_conley100, "model_interaction_land_degradation_average_slope.rds")
mf_conley_addInteractionLDs_addInteractionLDsAdapt_avg_slope <- read_model(model_dir_conley100, "model_interaction_land_degradation_adaptation_average_slope.rds")

get_total_assoc_unproc <- function(mf) {
  coef_tidy <- broom::tidy(mf) |>
    filter(term %in% TREATMENTS) |>
    select(term, estimate, std.error, p.value)

  # Sum estimated elasticity coefficients across degradation indicators
  # to obtain a combined association measure
  sum_coefs <- sum(coef_tidy$estimate)
  vcov_matrix <- vcov(mf)
  se_sum_coefs <- sqrt(sum(vcov_matrix[TREATMENTS, TREATMENTS]))
  lower_ci_sum_coefs <- sum_coefs - 1.96 * se_sum_coefs
  upper_ci_sum_coefs <- sum_coefs + 1.96 * se_sum_coefs
  z_stat <- sum_coefs / se_sum_coefs
  p_value_sum_coefs <- 2 * pnorm(-abs(z_stat))

  model_summary <- broom::glance(mf)
  Obs <- stats::nobs(mf)
  Adj_R2 <- if ("adj.r.squared" %in% colnames(model_summary)) model_summary$adj.r.squared else NA

  list(
    sum_coefs = sum_coefs,
    se_sum_coefs = se_sum_coefs,
    lower_ci_sum_coefs = lower_ci_sum_coefs,
    upper_ci_sum_coefs = upper_ci_sum_coefs,
    z_stat = z_stat,
    p_value_sum_coefs = p_value_sum_coefs,
    Obs = Obs,
    Adj_R2 = Adj_R2
  )
}

get_total_assoc_unproc_avgSlope <- function(mf) {
  # Sum estimated elasticity coefficients across degradation indicators
  # to obtain a combined association measure
  sum_coefs <- sum(mf$estimate)
  # The saved average-slope objects do not include the full covariance matrix
  # between treatment-specific average slopes. The summed average-slope
  # standard error is therefore computed from the treatment-specific standard
  # errors only.
  se_sum_coefs <- sqrt(sum(mf$std.error^2))
  lower_ci_sum_coefs <- sum_coefs - 1.96 * se_sum_coefs
  upper_ci_sum_coefs <- sum_coefs + 1.96 * se_sum_coefs
  z_stat <- sum_coefs / se_sum_coefs
  p_value_sum_coefs <- 2 * pnorm(-abs(z_stat))
  # Model-level fit summaries are not reported for these post-estimation
  # average-slope objects.
  Obs <- NA_integer_
  Adj_R2 <- NA_real_

  list(
    sum_coefs = sum_coefs,
    se_sum_coefs = se_sum_coefs,
    lower_ci_sum_coefs = lower_ci_sum_coefs,
    upper_ci_sum_coefs = upper_ci_sum_coefs,
    z_stat = z_stat,
    p_value_sum_coefs = p_value_sum_coefs,
    Obs = Obs,
    Adj_R2 = Adj_R2
  )
}

mf_noControls_total <- get_total_assoc_unproc(mf_noControls)
mf_naturalControls_total <- get_total_assoc_unproc(mf_naturalControls)
mf_naturalControls_adaptControls_total <- get_total_assoc_unproc(mf_naturalControls_adaptControls)
mf_naturalControls_adaptControls_socioeconControls_total <- get_total_assoc_unproc(mf_naturalControls_adaptControls_socioeconControls)
mf_addLongLat_total <- get_total_assoc_unproc(mf_addLongLat)
mf_addCropType_total <- get_total_assoc_unproc(mf_addCropType)
mf_rmSoilGrid_total <- get_total_assoc_unproc(mf_rmSoilGrid)
mf_winsor_total <- get_total_assoc_unproc(mf_winsor)
mf_addQuadraticLDs_avg_slope_total <- get_total_assoc_unproc_avgSlope(mf_addQuadraticLDs_avg_slope)
mf_addInteractionLDs_avg_slope_total <- get_total_assoc_unproc_avgSlope(mf_addInteractionLDs_avg_slope)
mf_addInteractionLDs_addInteractionLDsAdapt_avg_slope_total <- get_total_assoc_unproc_avgSlope(mf_addInteractionLDs_addInteractionLDsAdapt_avg_slope)
mf_HC1_total <- get_total_assoc_unproc(mf_HC1)
mf_Conley100_total <- get_total_assoc_unproc(mf_Conley100)

tidy_model_terms <- function(model, model_id) {
  broom::tidy(model, conf.int = FALSE) |>
    filter(term %in% TREATMENTS) |>
    mutate(
      conf.low = estimate - 1.96 * std.error,
      conf.high = estimate + 1.96 * std.error
    ) |>
    mutate(model = model_id) |>
    select(model, term, estimate, std.error, conf.low, conf.high, everything())
}

tidy_avg_slope_terms <- function(avg_slope_obj, model_id) {
  avg_slope_obj |>
    as_tibble() |>
    transmute(
      model = model_id,
      term = term,
      estimate = estimate,
      std.error = std.error,
      conf.low = estimate - 1.96 * std.error,
      conf.high = estimate + 1.96 * std.error
    ) |>
    filter(term %in% TREATMENTS)
}

finalize_model_order <- function(df, model_col = "model") {
  levels <- df |>
    distinct(.data[[model_col]]) |>
    mutate(.model_num = as.integer(as.character(.data[[model_col]]))) |>
    arrange(.model_num) |>
    pull(.data[[model_col]]) |>
    as.character()
  df |>
    mutate("{model_col}" := factor(.data[[model_col]], levels = levels))
}

plot_each_ld_terms <- function(df) {
  ggplot(
    df,
    aes(x = model, y = estimate, ymin = conf.low, ymax = conf.high, color = facet_label, shape = model_remark)
  ) +
    geom_point(size = 3, alpha = 1) +
    geom_errorbar(width = .0002) +
    geom_hline(yintercept = 0, linetype = "dotted", color = "black") +
    facet_wrap(~facet_label, nrow = 1) +
    theme_bw() +
    labs(x = "Model specification", y = "") +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 5)) +
    coord_cartesian(ylim = c(-0.15, 0.1)) +
    theme(strip.background = element_blank(), strip.text = element_text(face = "bold")) +
    theme(legend.position = "none") +
    scale_color_manual(values = c("#B40F20", "#FF0000", "#E58601", "#46ACC8", "#00A08A")) +
    scale_shape_manual(values = c(16, 1)) +
    theme(plot.margin = margin(t = 2, r = 2, b = 2, l = 2, unit = "pt")) +
    theme(axis.text.x = element_text(size = 7.5))
}

plot_total_terms <- function(df, vlines) {
  df |>
    ggplot(aes(x = model_id, y = sum_coefs, color = model_remark)) +
    geom_point(size = 4, alpha = 0.6, stroke = 2, shape = 16) +
    geom_errorbar(aes(ymin = lower_ci_sum_coefs, ymax = upper_ci_sum_coefs), height = .0002, width = .0002, size = 1.5) +
    geom_hline(yintercept = 0, linetype = "dotted", color = "black") +
    labs(
      x = "",
      y = "",
      title = "Elasticity of yield gaps to land degradation:\nPercent change in yield gaps\nassociated with a 1% increase in land degradation intensity"
    ) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
    coord_cartesian(ylim = c(-0.15, 0.25)) +
    theme_bw() +
    theme(legend.position = "none") +
    scale_colour_manual(values = c("#f8766d", "#00bfc4")) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    geom_vline(xintercept = vlines, linetype = "solid", color = "grey50") +
    theme(plot.margin = margin(t = 2, r = 2, b = 2, l = 2, unit = "pt")) +
    theme(axis.text.x = element_text(margin = margin(b = -15, unit = "pt"))) +
    theme(plot.title = element_text(hjust = 0.5))
}

combine_total_and_terms <- function(total_plot, term_plot) {
  plot_grid(
    total_plot,
    term_plot,
    labels = c("A", "B"),
    label_size = 12,
    label_x = 0,
    label_y = 1,
    nrow = 2,
    rel_heights = c(1.5, 1),
    scale = 1
  )
}

mf_all_fin <- bind_rows(
  tidy_model_terms(mf_noControls, "1"),
  tidy_model_terms(mf_naturalControls, "2"),
  tidy_model_terms(mf_naturalControls_adaptControls, "3"),
  tidy_model_terms(mf_naturalControls_adaptControls_socioeconControls, "4"),
  tidy_model_terms(mf_addLongLat, "5"),
  tidy_model_terms(mf_addCropType, "6"),
  tidy_model_terms(mf_rmSoilGrid, "7"),
  tidy_model_terms(mf_winsor, "8"),
  tidy_avg_slope_terms(mf_addQuadraticLDs_avg_slope, "9"),
  tidy_avg_slope_terms(mf_addInteractionLDs_avg_slope, "10"),
  tidy_avg_slope_terms(mf_addInteractionLDs_addInteractionLDsAdapt_avg_slope, "11"),
  tidy_model_terms(mf_HC1, "12"),
  tidy_model_terms(mf_Conley100, "13")
)

mf_all_fin <- mf_all_fin |>
  mutate(model_remark = if_else(model == "4", "Baseline", "Others"))

facet_label_tbl <- tribble(
  ~term, ~facet_label,
  "soilErosionDebt__rescaled__log", "(a)",
  "soilCompactionDebt__rescaled__log", "(b)",
  "soilOrganicCarbonDebt__rescaled__log", "(c)",
  "soilWaterDebt__rescaled__log", "(d)",
  "treeCoverDebt__rescaled__log", "(e)"
)

mf_all_fin <- mf_all_fin |>
  left_join(facet_label_tbl, by = "term") |>
  finalize_model_order()

plot_mf_all_fin <- plot_each_ld_terms(mf_all_fin)

mf_total_all_fin <- (as_tibble(mf_noControls_total) |> mutate(model = "mf_noControls_total", model_name_long = "(M) With no controls", model_id = "Model 1", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_naturalControls_total) |> mutate(model = "mf_naturalControls_total", model_name_long = "(M) With natural controls", model_id = "Model 2", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_naturalControls_adaptControls_total) |> mutate(model = "mf_naturalControls_adaptControls_total", model_name_long = "(M) With natural and management controls", model_id = "Model 3", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_naturalControls_adaptControls_socioeconControls_total) |> mutate(model = "mf_naturalControls_adaptControls_socioeconControls_total", model_name_long = "(M) With natural, management, and socio-economic-institutional controls", model_id = "Model 4", model_remark = "Baseline") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_addLongLat_total) |> mutate(model = "mf_addLongLat_total", model_name_long = "(M) With longitude and latitude controls", model_id = "Model 5", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_addCropType_total) |> mutate(model = "mf_addCropType_total", model_name_long = "(M) With crop type controls", model_id = "Model 6", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_rmSoilGrid_total) |> mutate(model = "mf_rmSoilGrid_total", model_name_long = "(M) Without soil continuous variables as controls", model_id = "Model 7", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_winsor_total) |> mutate(model = "mf_winsor_total", model_name_long = "(M) With winsorizing data to 2.5th and 97.5th quantiles", model_id = "Model 8", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_addQuadraticLDs_avg_slope_total) |> mutate(model = "mf_addQuadraticLDs_avg_slope_total", model_name_long = "(M) With quadratic terms of land degradation variables", model_id = "Model 9", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_addInteractionLDs_avg_slope_total) |> mutate(model = "mf_addInteractionLDs_avg_slope_total", model_name_long = "(M) With interaction terms among land degradation variables", model_id = "Model 10", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_addInteractionLDs_addInteractionLDsAdapt_avg_slope_total) |> mutate(model = "mf_addInteractionLDs_addInteractionLDsAdapt_avg_slope_total", model_name_long = "(M) With interaction terms among land degradation variables, and interaction terms between land degradation variables and management variables", model_id = "Model 11", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_HC1_total) |> mutate(model = "mf_HC1_total", model_name_long = "(M) With HC1 standard error", model_id = "Model 12", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_Conley100_total) |> mutate(model = "mf_Conley100_total", model_name_long = "(M) With Conley standard error (cutoff 100-km)", model_id = "Model 13", model_remark = "Others") |> select(model, everything())) |>
  mutate(
    model_id = factor(model_id, levels = model_id),
    model_remark = factor(model_remark, levels = c("Baseline", "Others"))
  )

plot_mf_total_all_fin <- plot_total_terms(
  mf_total_all_fin,
  c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
)

combined_plot <- combine_total_and_terms(plot_mf_total_all_fin, plot_mf_all_fin)

saveRDS(
  list(
    mf_total_all_fin = mf_total_all_fin,
    mf_all_fin = mf_all_fin,
    input_files = required_files
  ),
  path_figure_data("Fig2Data_CoefPlotModelSpecifications.rds")
)
write_csv(mf_total_all_fin, path_figure_data("Fig2Data_CoefPlotModelSpecificationsTotal.csv"))
write_csv(mf_all_fin, path_figure_data("Fig2Data_CoefPlotModelSpecificationsIndividual.csv"))


mf_addFieldSize_total <- get_total_assoc_unproc(mf_addFieldSize)
mf_addSoilDepth_total <- get_total_assoc_unproc(mf_addSoilDepth)
mf_addSoilN_total <- get_total_assoc_unproc(mf_addSoilN)
mf_addWTD_total <- get_total_assoc_unproc(mf_addWTD)
mf_addRoadDensity_total <- get_total_assoc_unproc(mf_addRoadDensity)
mf_addTravelCities_total <- get_total_assoc_unproc(mf_addTravelCities)
mf_rmMachine_total <- get_total_assoc_unproc(mf_rmMachine)
mf_addAgriExpend_total <- get_total_assoc_unproc(mf_addAgriExpend)
mf_addEconFree_total <- get_total_assoc_unproc(mf_addEconFree)

mf_total_othersForSI_fin <- (as_tibble(mf_naturalControls_adaptControls_socioeconControls_total) |> mutate(model = "mf_naturalControls_adaptControls_socioeconControls_total", model_id = "Model 4", model_remark = "Baseline") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_addFieldSize_total) |> mutate(model = "mf_addFieldSize_total", model_id = "Model 14", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_addSoilDepth_total) |> mutate(model = "mf_addSoilDepth_total", model_id = "Model 15", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_addSoilN_total) |> mutate(model = "mf_addSoilN_total", model_id = "Model 16", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_addWTD_total) |> mutate(model = "mf_addWTD_total", model_id = "Model 17", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_addRoadDensity_total) |> mutate(model = "mf_addRoadDensity_total", model_id = "Model 18", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_addTravelCities_total) |> mutate(model = "mf_addTravelCities_total", model_id = "Model 19", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_rmMachine_total) |> mutate(model = "mf_rmMachine_total", model_id = "Model 20", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_addAgriExpend_total) |> mutate(model = "mf_addAgriExpend_total", model_id = "Model 21", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_addEconFree_total) |> mutate(model = "mf_addEconFree_total", model_id = "Model 22", model_remark = "Others") |> select(model, everything())) |>
  mutate(
    model_id = factor(model_id, levels = model_id),
    model_remark = factor(model_remark, levels = c("Baseline", "Others"))
  )

mf_othersForSI_fin <- bind_rows(
  tidy_model_terms(mf_naturalControls_adaptControls_socioeconControls, "4"),
  tidy_model_terms(mf_addFieldSize, "14"),
  tidy_model_terms(mf_addSoilDepth, "15"),
  tidy_model_terms(mf_addSoilN, "16"),
  tidy_model_terms(mf_addWTD, "17"),
  tidy_model_terms(mf_addRoadDensity, "18"),
  tidy_model_terms(mf_addTravelCities, "19"),
  tidy_model_terms(mf_rmMachine, "20"),
  tidy_model_terms(mf_addAgriExpend, "21"),
  tidy_model_terms(mf_addEconFree, "22")
) |>
  mutate(model_remark = if_else(as.character(model) == "4", "Baseline", "Others")) |>
  left_join(facet_label_tbl, by = "term") |>
  finalize_model_order()

plot_mf_total_othersForSI_fin <- plot_total_terms(
  mf_total_othersForSI_fin,
  c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5)
)
plot_mf_othersForSI_fin <- plot_each_ld_terms(mf_othersForSI_fin)
combined_plot_others <- combine_total_and_terms(plot_mf_total_othersForSI_fin, plot_mf_othersForSI_fin)

saveRDS(
  list(
    mf_total_othersForSI_fin = mf_total_othersForSI_fin,
    mf_othersForSI_fin = mf_othersForSI_fin,
    input_files = required_files
  ),
  path_figure_data("SuppFig3Data_CoefPlotRobustnessModelSpecifications.rds")
)
write_csv(mf_total_othersForSI_fin, path_figure_data("SuppFig3Data_CoefPlotRobustnessModelSpecificationsTotal.csv"))
write_csv(mf_othersForSI_fin, path_figure_data("SuppFig3Data_CoefPlotRobustnessModelSpecificationsIndividual.csv"))


mf_conley_noControls_total <- get_total_assoc_unproc(mf_conley_noControls)
mf_conley_naturalControls_total <- get_total_assoc_unproc(mf_conley_naturalControls)
mf_conley_naturalControls_adaptControls_total <- get_total_assoc_unproc(mf_conley_naturalControls_adaptControls)
mf_conley_naturalControls_adaptControls_socioeconControls_total <- get_total_assoc_unproc(mf_conley_naturalControls_adaptControls_socioeconControls)
mf_conley_addLongLat_total <- get_total_assoc_unproc(mf_conley_addLongLat)
mf_conley_addCropType_total <- get_total_assoc_unproc(mf_conley_addCropType)
mf_conley_rmSoilGrid_total <- get_total_assoc_unproc(mf_conley_rmSoilGrid)
mf_conley_winsor_total <- get_total_assoc_unproc(mf_conley_winsor)
mf_conley_addQuadraticLDs_avg_slope_total <- get_total_assoc_unproc_avgSlope(mf_conley_addQuadraticLDs_avg_slope)
mf_conley_addInteractionLDs_avg_slope_total <- get_total_assoc_unproc_avgSlope(mf_conley_addInteractionLDs_avg_slope)
mf_conley_addInteractionLDs_addInteractionLDsAdapt_avg_slope_total <- get_total_assoc_unproc_avgSlope(mf_conley_addInteractionLDs_addInteractionLDsAdapt_avg_slope)

mf_conley_all_fin <- bind_rows(
  tidy_model_terms(mf_conley_noControls, "1"),
  tidy_model_terms(mf_conley_naturalControls, "2"),
  tidy_model_terms(mf_conley_naturalControls_adaptControls, "3"),
  tidy_model_terms(mf_conley_naturalControls_adaptControls_socioeconControls, "4"),
  tidy_model_terms(mf_conley_addLongLat, "5"),
  tidy_model_terms(mf_conley_addCropType, "6"),
  tidy_model_terms(mf_conley_rmSoilGrid, "7"),
  tidy_model_terms(mf_conley_winsor, "8"),
  tidy_avg_slope_terms(mf_conley_addQuadraticLDs_avg_slope, "9"),
  tidy_avg_slope_terms(mf_conley_addInteractionLDs_avg_slope, "10"),
  tidy_avg_slope_terms(mf_conley_addInteractionLDs_addInteractionLDsAdapt_avg_slope, "11")
) |>
  mutate(model_remark = if_else(as.character(model) == "4", "Baseline", "Others")) |>
  left_join(facet_label_tbl, by = "term") |>
  finalize_model_order()

mf_conley_total_all_fin <- (as_tibble(mf_conley_noControls_total) |> mutate(model = "mf_noControls_total", model_name_long = "(M) With no controls", model_id = "Model 1", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_conley_naturalControls_total) |> mutate(model = "mf_naturalControls_total", model_name_long = "(M) With natural controls", model_id = "Model 2", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_conley_naturalControls_adaptControls_total) |> mutate(model = "mf_naturalControls_adaptControls_total", model_name_long = "(M) With natural and management controls", model_id = "Model 3", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_conley_naturalControls_adaptControls_socioeconControls_total) |> mutate(model = "mf_naturalControls_adaptControls_socioeconControls_total", model_name_long = "(M) With natural, management, and socio-economic-institutional controls", model_id = "Model 4", model_remark = "Baseline") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_conley_addLongLat_total) |> mutate(model = "mf_addLongLat_total", model_name_long = "(M) With longitude and latitude controls", model_id = "Model 5", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_conley_addCropType_total) |> mutate(model = "mf_addCropType_total", model_name_long = "(M) With crop type controls", model_id = "Model 6", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_conley_rmSoilGrid_total) |> mutate(model = "mf_rmSoilGrid_total", model_name_long = "(M) Without soil continuous variables as controls", model_id = "Model 7", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_conley_winsor_total) |> mutate(model = "mf_winsor_total", model_name_long = "(M) With winsorizing data to 2.5th and 97.5th quantiles", model_id = "Model 8", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_conley_addQuadraticLDs_avg_slope_total) |> mutate(model = "mf_addQuadraticLDs_avg_slope_total", model_name_long = "(M) With quadratic terms of land degradation variables", model_id = "Model 9", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_conley_addInteractionLDs_avg_slope_total) |> mutate(model = "mf_addInteractionLDs_avg_slope_total", model_name_long = "(M) With interaction terms among land degradation variables", model_id = "Model 10", model_remark = "Others") |> select(model, everything())) |>
  bind_rows(as_tibble(mf_conley_addInteractionLDs_addInteractionLDsAdapt_avg_slope_total) |> mutate(model = "mf_addInteractionLDs_addInteractionLDsAdapt_avg_slope_total", model_name_long = "(M) With interaction terms among land degradation variables, and interaction terms between land degradation variables and management variables", model_id = "Model 11", model_remark = "Others") |> select(model, everything())) |>
  mutate(
    model_id = factor(model_id, levels = model_id),
    model_remark = factor(model_remark, levels = c("Baseline", "Others"))
  )

plot_mf_conley_total_all_fin <- plot_total_terms(
  mf_conley_total_all_fin,
  c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5)
)
plot_mf_conley_all_fin <- plot_each_ld_terms(mf_conley_all_fin)
combined_plot_conley <- combine_total_and_terms(plot_mf_conley_total_all_fin, plot_mf_conley_all_fin)

saveRDS(
  list(
    mf_conley_total_all_fin = mf_conley_total_all_fin,
    mf_conley_all_fin = mf_conley_all_fin,
    input_files = required_files[grepl("Conley100", required_files)]
  ),
  path_figure_data("SuppFig2Data_CoefPlotConleyModelSpecifications.rds")
)
write_csv(mf_conley_total_all_fin, path_figure_data("SuppFig2Data_CoefPlotConleyModelSpecificationsTotal.csv"))
write_csv(mf_conley_all_fin, path_figure_data("SuppFig2Data_CoefPlotConleyModelSpecificationsIndividual.csv"))



