# Generate Supplementary Fig. 4: residualized-treatment robustness check.

library(ggplot2)

source(file.path("code", "_shared", "paths.R"))

coef_data <- readRDS(path_figure_data("SuppFig4Data_CoefPlotResidualizedTreatmentsRobustness.rds"))$coef_data

p <- ggplot(coef_data, aes(x = estimate, y = term_clean, color = model)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_pointrange(
    aes(xmin = conf.low, xmax = conf.high),
    position = position_dodge(width = 0.5)
  ) +
  labs(x = "Coefficient Estimate", y = "", color = "Model") +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave(path_figure("SuppFig4_CoefPlotResidualizedTreatmentsRobustness.pdf"), plot = p, width = 6, height = 4)
ggsave(path_figure("SuppFig4_CoefPlotResidualizedTreatmentsRobustness.png"), plot = p, width = 6, height = 4, dpi = 300)
