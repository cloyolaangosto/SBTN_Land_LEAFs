# Generate Main Fig. 3: yield-gap and actual-yield coefficient comparison.
#
# The figure compares land-degradation coefficients when the outcome is crop
# yield gaps versus Ray/Gerber actual yields. It reads model objects produced by
# the regression script.

library(broom)
library(cowplot)
library(dplyr)
library(ggplot2)
library(readr)

source(file.path("code", "_shared", "paths.R"))

TREATMENTS <- c(
  "soilErosionDebt__rescaled__log",
  "soilCompactionDebt__rescaled__log",
  "soilOrganicCarbonDebt__rescaled__log",
  "soilWaterDebt__rescaled__log",
  "treeCoverDebt__rescaled__log"
)

coef_rename_tbl <- c(
  "soilErosionDebt__rescaled__log" = "Soil erosion debt",
  "soilCompactionDebt__rescaled__log" = "Soil compaction debt",
  "soilOrganicCarbonDebt__rescaled__log" = "Soil organic carbon debt",
  "soilWaterDebt__rescaled__log" = "Soil water debt",
  "treeCoverDebt__rescaled__log" = "Tree cover debt"
)

model_dir <- path_processed("models", "main_10km")

model_paths <- c(
  yield_gap = file.path(model_dir, "model_baseline.rds"),
  actual_yield = file.path(model_dir, "model_actual_yield.rds")
)

assert_files_exist(model_paths)

mf <- readRDS(model_paths[["yield_gap"]])
mf_actualYieldGerberRay <- readRDS(model_paths[["actual_yield"]])

calc_total <- function(model) {
  # Sum estimated elasticity coefficients across degradation indicators
  # to obtain a combined association measure
  s <- sum(coef(model)[TREATMENTS])
  v <- vcov(model)
  se <- sqrt(sum(v[TREATMENTS, TREATMENTS]))
  tibble(
    estimate = s,
    std.error = se,
    conf.low = s - 1.96 * se,
    conf.high = s + 1.96 * se
  )
}

total_yieldgap <- calc_total(mf)
total_actualYield <- calc_total(mf_actualYieldGerberRay)

coef_data <- bind_rows(
  tidy(mf, conf.int = TRUE) |> mutate(model = "Yield gap"),
  tidy(mf_actualYieldGerberRay, conf.int = TRUE) |> mutate(model = "Actual yield")
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
    term_clean = factor(term_clean, levels = unname(coef_rename_tbl))
  )

all_levels <- c(unname(coef_rename_tbl), "Overall")

total_effects <- bind_rows(
  total_yieldgap |> mutate(model = "Yield gap"),
  total_actualYield |> mutate(model = "Actual yield")
) |>
  mutate(
    term = "Overall",
    term_clean = factor("Overall", levels = all_levels)
  )

coef_data_with_total <- bind_rows(
  coef_data |> mutate(term_clean = factor(term_clean, levels = all_levels)),
  total_effects
)

panel_theme <- list(
  theme_bw(),
  theme(
    strip.background = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "none",
    plot.margin = margin(t = 2, r = 2, b = 2, l = 2, unit = "pt"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 7.5)
  ),
  scale_color_manual(values = c("#B40F20", "#FF0000", "#E58601", "#46ACC8", "#00A08A", "#333333"))
)

p1 <- coef_data_with_total |>
  filter(model == "Yield gap") |>
  ggplot(aes(x = term_clean, y = estimate, ymin = conf.low, ymax = conf.high, color = term_clean)) +
  geom_point(size = 3) +
  geom_errorbar(width = 0.2) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "black") +
  labs(x = "", y = "Change in yield gaps (%)") +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 5), limits = c(0, 0.2)) +
  panel_theme

p2 <- coef_data_with_total |>
  filter(model == "Actual yield") |>
  ggplot(aes(x = term_clean, y = estimate, ymin = conf.low, ymax = conf.high, color = term_clean)) +
  geom_point(size = 3) +
  geom_errorbar(width = 0.2) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "black") +
  labs(x = "", y = "Change in actual yield (%)") +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 5), limits = c(-0.2, 0)) +
  panel_theme

combined_plot <- plot_grid(
  p1,
  p2,
  labels = c("A", "B"),
  label_size = 12,
  label_x = 0,
  label_y = 1,
  nrow = 1,
  scale = 1
)

saveRDS(
  list(coef_data_with_total = coef_data_with_total, input_files = model_paths),
  path_figure_data("Fig3Data_CoefPlotYieldGapActualYield.rds")
)
write_csv(
  coef_data_with_total,
  path_figure_data("Fig3Data_CoefPlotYieldGapActualYield.csv")
)


