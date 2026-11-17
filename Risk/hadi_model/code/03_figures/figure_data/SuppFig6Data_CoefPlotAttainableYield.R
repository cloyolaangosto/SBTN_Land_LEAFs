# Generate Supplementary Fig. 6: attainable-yield coefficient plot.
#
# The figure reports land-degradation coefficients from the attainable-yield
# outcome specification using the saved regression model object.

library(broom)
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

all_levels <- c(
  "Soil erosion debt",
  "Soil compaction debt",
  "Soil organic carbon debt",
  "Soil water debt",
  "Tree cover debt",
  "Overall"
)

model_path <- path_processed(
  "models",
  "main_10km",
  "model_attainable_yield.rds"
)
assert_files_exist(model_path)

mf_attainYieldGerberRay <- readRDS(model_path)

# Sum estimated elasticity coefficients across degradation indicators
# to obtain a combined association measure
sum_attain <- sum(coef(mf_attainYieldGerberRay)[TREATMENTS])
vcov_attain <- vcov(mf_attainYieldGerberRay)
se_attain <- sqrt(sum(vcov_attain[TREATMENTS, TREATMENTS]))

coef_attain <- tidy(mf_attainYieldGerberRay, conf.int = TRUE) |>
  filter(term %in% TREATMENTS) |>
  mutate(
    term_clean = case_when(
      term == "soilErosionDebt__rescaled__log" ~ "Soil erosion debt",
      term == "soilCompactionDebt__rescaled__log" ~ "Soil compaction debt",
      term == "soilOrganicCarbonDebt__rescaled__log" ~ "Soil organic carbon debt",
      term == "soilWaterDebt__rescaled__log" ~ "Soil water debt",
      term == "treeCoverDebt__rescaled__log" ~ "Tree cover debt",
      TRUE ~ term
    )
  )

total_attain <- tibble(
  term = "Overall",
  estimate = sum_attain,
  std.error = se_attain,
  conf.low = sum_attain - 1.96 * se_attain,
  conf.high = sum_attain + 1.96 * se_attain,
  term_clean = "Overall"
)

coef_attain_with_total <- bind_rows(
  coef_attain |> mutate(term_clean = factor(term_clean, levels = all_levels)),
  total_attain |> mutate(term_clean = factor(term_clean, levels = all_levels))
)

p_attain <- ggplot(
  coef_attain_with_total,
  aes(x = term_clean, y = estimate, ymin = conf.low, ymax = conf.high, color = term_clean)
) +
  geom_point(size = 3) +
  geom_errorbar(width = 0.2) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "black") +
  labs(x = "", y = "Change in attainable yield (%)") +
  scale_y_continuous(breaks = seq(-0.2, 0.01, by = 0.05), limits = c(-0.20, 0.01)) +
  theme_bw() +
  theme(
    legend.position = "none",
    plot.margin = margin(t = 2, r = 2, b = 2, l = 2, unit = "pt"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 7.5)
  ) +
  scale_color_manual(values = c("#B40F20", "#FF0000", "#E58601", "#46ACC8", "#00A08A", "#333333"))

saveRDS(
  list(coef_attain_with_total = coef_attain_with_total, input_files = c(attainable_yield_model = model_path)),
  path_figure_data("SuppFig6Data_CoefPlotAttainableYield.rds")
)
write_csv(coef_attain_with_total, path_figure_data("SuppFig6Data_CoefPlotAttainableYield.csv"))

