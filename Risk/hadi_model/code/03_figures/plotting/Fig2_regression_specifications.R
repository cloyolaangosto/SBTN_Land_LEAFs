# Generate Main Fig. 2, Supplementary Fig. 2, and Supplementary Fig. 3.
#
# The plotting inputs are the saved coefficient-plot figure-data objects created
# by `code/03_figures/figure_data/Fig2Data_CoefPlotModelSpecifications.R`.

library(cowplot)
library(dplyr)
library(ggplot2)

source(file.path("code", "_shared", "paths.R"))

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

plot_and_save_coef_figure <- function(data_file, total_name, individual_name, output_basename, vlines) {
  fig_data <- readRDS(path_figure_data(data_file))
  total_data <- fig_data[[total_name]]
  individual_data <- fig_data[[individual_name]]

  combined_plot <- combine_total_and_terms(
    plot_total_terms(total_data, vlines),
    plot_each_ld_terms(individual_data)
  )

  ggsave(
    path_figure(paste0(output_basename, ".png")),
    combined_plot,
    width = 18,
    height = 14,
    unit = "cm",
    dpi = 300
  )
  ggsave(
    path_figure(paste0(output_basename, ".pdf")),
    combined_plot,
    width = 18,
    height = 14,
    unit = "cm"
  )
}

plot_and_save_coef_figure(
  "Fig2Data_CoefPlotModelSpecifications.rds",
  "mf_total_all_fin",
  "mf_all_fin",
  "Fig2_CoefPlotModelSpecifications",
  c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
)

plot_and_save_coef_figure(
  "SuppFig3Data_CoefPlotRobustnessModelSpecifications.rds",
  "mf_total_othersForSI_fin",
  "mf_othersForSI_fin",
  "SuppFig3_CoefPlotRobustnessModelSpecifications",
  c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5)
)

plot_and_save_coef_figure(
  "SuppFig2Data_CoefPlotConleyModelSpecifications.rds",
  "mf_conley_total_all_fin",
  "mf_conley_all_fin",
  "SuppFig2_CoefPlotConleyModelSpecifications",
  c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5)
)
