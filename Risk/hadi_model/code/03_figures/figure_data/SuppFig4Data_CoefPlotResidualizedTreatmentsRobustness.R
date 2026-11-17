# Generate Supplementary Fig. 4: residualized-treatment robustness check.
#
# The figure compares the baseline coefficients with a specification where
# land-degradation variables are residualized against the control variables. It
# does not run average-slope calculations.

library(broom)
library(dplyr)
library(fixest)
library(ggplot2)
library(readr)
library(tidyr)

source(file.path("code", "_shared", "paths.R"))

dataset_path <- path_processed("final_dataset", "analysis_ready_data_full.rds")
model_dir <- path_processed("models", "main_10km")
model_var_path <- file.path(model_dir, "model_spec_variables.rds")
baseline_model_path <- file.path(model_dir, "model_baseline.rds")

assert_files_exist(c(dataset_path, model_var_path, baseline_model_path))

model_vars <- readRDS(model_var_path)
OUTCOME <- model_vars$OUTCOME
TREATMENTS <- model_vars$TREATMENTS
CONTROLS_environment_climateWc <- model_vars$CONTROLS_environment_climateWc
CONTROLS_environment_climateTerra <- model_vars$CONTROLS_environment_climateTerra
CONTROLS_environment_timeDelta_climateTerra <- model_vars$CONTROLS_environment_timeDelta_climateTerra
CONTROLS_environment_terrain <- model_vars$CONTROLS_environment_terrain
CONTROLS_environment_soilType <- model_vars$CONTROLS_environment_soilType
CONTROLS_environment_soilGrid <- model_vars$CONTROLS_environment_soilGrid
CONTROLS_adaptation <- model_vars$CONTROLS_adaptation
CONTROLS_socioeconomic <- model_vars$CONTROLS_socioeconomic

coef_rename_tbl <- c(
  "soilErosionDebt__rescaled__log" = "Soil erosion debt",
  "soilCompactionDebt__rescaled__log" = "Soil compaction debt",
  "soilOrganicCarbonDebt__rescaled__log" = "Soil organic carbon debt",
  "soilWaterDebt__rescaled__log" = "Soil water debt",
  "treeCoverDebt__rescaled__log" = "Tree cover debt"
)

df_ok <- readRDS(dataset_path)
mf <- readRDS(baseline_model_path)

df_ok_noNA <- df_ok |>
  select(all_of(c(
    OUTCOME,
    TREATMENTS,
    CONTROLS_environment_climateWc,
    CONTROLS_environment_climateTerra,
    CONTROLS_environment_timeDelta_climateTerra,
    CONTROLS_environment_terrain,
    CONTROLS_environment_soilType,
    CONTROLS_environment_soilGrid,
    CONTROLS_adaptation,
    CONTROLS_socioeconomic
  ))) |>
  drop_na()

controls_vars <- c(
  CONTROLS_environment_climateWc,
  CONTROLS_environment_climateTerra,
  CONTROLS_environment_timeDelta_climateTerra,
  CONTROLS_environment_terrain,
  CONTROLS_environment_soilType,
  CONTROLS_environment_soilGrid,
  CONTROLS_adaptation
)

res_treatments <- sapply(TREATMENTS, function(treat) {
  f_t <- as.formula(paste(treat, "~", paste(controls_vars, collapse = " + ")))
  lm(f_t, data = df_ok_noNA)$residuals
})
res_treatments <- as.data.frame(res_treatments)
colnames(res_treatments) <- TREATMENTS

df_residOnlyT <- cbind.data.frame(
  AllCrops2010YieldGapPercent__rescaled__log = df_ok_noNA[[OUTCOME]],
  res_treatments,
  df_ok_noNA |> select(all_of(CONTROLS_socioeconomic))
)

f_residOnlyT <- as.formula(
  paste(OUTCOME, "~", paste(c(TREATMENTS, CONTROLS_socioeconomic), collapse = " + "))
)
m_residOnlyT <- fixest::feols(f_residOnlyT, data = df_residOnlyT, vcov = "hetero")

coef_data <- bind_rows(
  tidy(mf, conf.int = TRUE) |> mutate(model = "Baseline"),
  tidy(m_residOnlyT, conf.int = TRUE) |> mutate(model = "Residualized Treatment")
) |>
  filter(term %in% TREATMENTS) |>
  mutate(
    term_clean = case_when(
      term == "soilErosionDebt__rescaled__log" ~ "Soil erosion debt",
      term == "soilCompactionDebt__rescaled__log" ~ "Soil compaction debt",
      term == "soilOrganicCarbonDebt__rescaled__log" ~ "Soil organic carbon debt",
      term == "soilWaterDebt__rescaled__log" ~ "Soil water debt",
      term == "treeCoverDebt__rescaled__log" ~ "Tree cover debt",
      TRUE ~ term
    ),
    term_clean = factor(term_clean, levels = rev(unname(coef_rename_tbl)))
  )

p <- ggplot(coef_data, aes(x = estimate, y = term_clean, color = model)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_pointrange(
    aes(xmin = conf.low, xmax = conf.high),
    position = position_dodge(width = 0.5)
  ) +
  labs(x = "Coefficient Estimate", y = "", color = "Model") +
  theme_minimal() +
  theme(legend.position = "bottom")

saveRDS(
  list(
    model = m_residOnlyT,
    coef_data = coef_data,
    input_files = c(dataset = dataset_path, model_vars = model_var_path, baseline_model = baseline_model_path)
  ),
  path_figure_data("SuppFig4Data_CoefPlotResidualizedTreatmentsRobustness.rds")
)
write_csv(coef_data, path_figure_data("SuppFig4Data_CoefPlotResidualizedTreatmentsRobustness.csv"))


