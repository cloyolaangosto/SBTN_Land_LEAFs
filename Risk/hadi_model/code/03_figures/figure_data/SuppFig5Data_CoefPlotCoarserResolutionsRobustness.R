# Generate Supplementary Fig. 5: spatial-resolution robustness check.
#
# This script fits the 50 km and 100 km robustness regressions from saved
# aggregated data and compares them with the 10 km baseline. 


library(broom)
library(dplyr)
library(fixest)
library(ggplot2)
library(readr)

source(file.path("code", "_shared", "paths.R"))

OUTCOME <- "AllCrops2010YieldGapPercent__rescaled__log"

TREATMENTS <- c(
  "soilErosionDebt__rescaled__log",
  "soilCompactionDebt__rescaled__log",
  "soilOrganicCarbonDebt__rescaled__log",
  "soilWaterDebt__rescaled__log",
  "treeCoverDebt__rescaled__log"
)

CONTROLS_environment_timeDelta_climateTerra <- c(
  "pr_sum_mean__mean_delta__rescaled__log",
  "tmmn_mean_mean__mean_delta__rescaled__log",
  "tmmx_mean_mean__mean_delta__rescaled__log",
  "srad_mean_mean__mean_delta__rescaled__log"
)

CONTROLS_environment_climateTerra <- c(
  "pr_end_sum_mean__mean__rescaled__log",
  "tmmn_end_mean_mean__mean__rescaled__log",
  "tmmx_end_mean_mean__mean__rescaled__log",
  "srad_end_mean_mean__mean__rescaled__log"
)

CONTROLS_environment_terrain <- c(
  "gmted_elevation__mean__rescaled__log",
  "gmted_slope__mean__rescaled__log",
  "geomorpho90m_tri__mean__rescaled__log"
)

CONTROLS_environment_soilType <- c("soil_type__mode")

CONTROLS_environment_soilGrid <- c(
  "clay__mean__rescaled__log",
  "sand__mean__rescaled__log",
  "silt__mean__rescaled__log"
)

CONTROLS_environment_climateWc <- c(
  "wc_srad__mean__rescaled__log",
  "wc_tavg__mean__rescaled__log",
  "wc_prec__mean__rescaled__log",
  "wc_meanTempWarmestQuarter__mean__rescaled__log",
  "wc_maxTempWarmestMonth__mean__rescaled__log",
  "wc_precWettestQuarter__mean__rescaled__log",
  "aridity__mean__rescaled__log"
)

CONTROLS_adaptation <- c(
  "nfer_crop_nh4_end_mean__mean__rescaled__log",
  "nfer_crop_no3_end_mean__mean__rescaled__log",
  "nmanure_app_crop_end_mean__mean__rescaled__log",
  "ndep_noy_end_mean__mean__rescaled__log",
  "ndep_nhx_end_mean__mean__rescaled__log",
  "phosphorus_OlsenP_kgha1_World_Aug2022_ver__mean__rescaled__log",
  "pesticide_APR_sum_2015_H__mean__rescaled__log",
  "irrigation_AEI_ha_end__mean__rescaled__log"
)

CONTROLS_socioeconomic <- c(
  "hdi_ca2010__mean__rescaled__log",
  "gdp_ppp_ca2010__mean__rescaled__log",
  "gdp_per_capita_ca2010__mean__rescaled__log",
  "property_rights_protection__rescaled__log",
  "enforcement_imputed__rescaled__log",
  "electricityRural_ca2010__rescaled__log",
  "phoneSub_ca2010__rescaled__log",
  "agriShr_ca2010__rescaled__log",
  "agriEmp_ca2010__rescaled__log",
  "bayesian_corruption__rescaled__log",
  "machinery_ca2010__rescaled__log"
)

f <- paste(
  OUTCOME,
  "~",
  paste(TREATMENTS, collapse = " + "), "+",
  paste(c(
    CONTROLS_environment_climateWc,
    CONTROLS_environment_climateTerra,
    CONTROLS_environment_timeDelta_climateTerra,
    CONTROLS_environment_terrain,
    CONTROLS_environment_soilType,
    CONTROLS_environment_soilGrid
  ), collapse = " + "), "+",
  paste(CONTROLS_adaptation, collapse = " + "), "+",
  paste(CONTROLS_socioeconomic, collapse = " + ")
)
f <- as.formula(f)

vcov <- NULL

coef_rename_tbl <- c(
  "soilErosionDebt__rescaled__log" = "Soil erosion debt",
  "soilCompactionDebt__rescaled__log" = "Soil compaction debt",
  "soilOrganicCarbonDebt__rescaled__log" = "Soil organic carbon debt",
  "soilWaterDebt__rescaled__log" = "Soil water debt",
  "treeCoverDebt__rescaled__log" = "Tree cover debt"
)

baseline_model_path <- path_processed(
  "models",
  "main_10km",
  "model_baseline.rds"
)

data_50km_path <- path_intermediate(
  "coarser_resolution",
  "loglog",
  "analysis_ready_data_50km.rds"
)

data_100km_path <- path_intermediate(
  "coarser_resolution",
  "loglog",
  "analysis_ready_data_100km.rds"
)

assert_files_exist(c(baseline_model_path, data_50km_path, data_100km_path))

mf <- readRDS(baseline_model_path)

df_logLog_50Km <- readRDS(data_50km_path) |>
  mutate(lat = y, long = x)

mf_50Km <- fixest::feols(
  f,
  data = df_logLog_50Km,
  vcov = vcov
)

df_logLog_100Km <- readRDS(data_100km_path) |>
  mutate(lat = y, long = x)

mf_100Km <- fixest::feols(
  f,
  data = df_logLog_100Km,
  vcov = vcov
)

coef_data <- bind_rows(
  tidy(mf, conf.int = TRUE) |> mutate(model = "Baseline (10 km)"),
  tidy(mf_50Km, conf.int = TRUE) |> mutate(model = "50 km"),
  tidy(mf_100Km, conf.int = TRUE) |> mutate(model = "100 km")
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
    term_clean = factor(term_clean, levels = rev(unname(coef_rename_tbl))),
    model = factor(model, levels = c("Baseline (10 km)", "50 km", "100 km"))
  )

p <- ggplot(coef_data, aes(x = estimate, y = term_clean, color = model)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_pointrange(
    aes(xmin = conf.low, xmax = conf.high),
    position = position_dodge(width = 0.5)
  ) +
  labs(
    x = "Coefficient Estimate",
    y = "",
    color = "Model"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

saveRDS(
  list(
    coef_data = coef_data,
    formula = deparse(f),
    input_files = c(
      baseline_model = baseline_model_path,
      data_50km = data_50km_path,
      data_100km = data_100km_path
    )
  ),
  path_figure_data("SuppFig5Data_CoefPlotCoarserResolutionsRobustness.rds")
)
write_csv(
  coef_data,
  path_figure_data("SuppFig5Data_CoefPlotCoarserResolutionsRobustness.csv")
)

