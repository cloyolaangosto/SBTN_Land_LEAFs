# Generate Supplementary Fig. 5: spatial-resolution robustness check.

library(ggplot2)

source(file.path("code", "_shared", "paths.R"))

coef_data <- readRDS(path_figure_data("SuppFig5Data_CoefPlotCoarserResolutionsRobustness.rds"))$coef_data

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

ggsave(path_figure("SuppFig5_CoefPlotCoarserResolutionsRobustness.pdf"), plot = p, width = 6, height = 4)
ggsave(path_figure("SuppFig5_CoefPlotCoarserResolutionsRobustness.png"), plot = p, width = 6, height = 4, dpi = 300)
