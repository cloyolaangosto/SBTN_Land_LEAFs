# Generate Main Fig. 3: yield-gap and actual-yield coefficient comparison.

library(cowplot)
library(dplyr)
library(ggplot2)

source(file.path("code", "_shared", "paths.R"))

coef_data_with_total <- readRDS(path_figure_data("Fig3Data_CoefPlotYieldGapActualYield.rds"))$coef_data_with_total

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

ggsave(
  path_figure("Fig3_CoefPlotYieldGapActualYield.png"),
  combined_plot,
  width = 18,
  height = 9,
  unit = "cm",
  dpi = 300
)

ggsave(
  path_figure("Fig3_CoefPlotYieldGapActualYield.pdf"),
  combined_plot,
  width = 18,
  height = 9,
  unit = "cm"
)
