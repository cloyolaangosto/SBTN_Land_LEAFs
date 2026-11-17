# Generate Supplementary Fig. 6: attainable-yield coefficient plot.

library(ggplot2)

source(file.path("code", "_shared", "paths.R"))

coef_attain_with_total <- readRDS(path_figure_data("SuppFig6Data_CoefPlotAttainableYield.rds"))$coef_attain_with_total

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

ggsave(path_figure("SuppFig6_CoefPlotAttainableYield.png"), p_attain, width = 9, height = 9, unit = "cm", dpi = 300)
ggsave(path_figure("SuppFig6_CoefPlotAttainableYield.pdf"), p_attain, width = 9, height = 9, unit = "cm")
