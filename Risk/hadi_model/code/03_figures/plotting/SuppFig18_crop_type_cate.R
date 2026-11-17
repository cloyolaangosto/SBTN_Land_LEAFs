# Generate Supplementary Fig. 18: CATE heterogeneity by dominant crop type.

library(cowplot)
library(dplyr)
library(ggplot2)

source(file.path("code", "_shared", "paths.R"))

fig_data <- readRDS(path_figure_data("SuppFig18Data_CateByCropType.rds"))
df_sum_positive <- fig_data$data
df_long <- fig_data$process_data

make_boxplot <- function(data, title, y_limits) {
  ggplot(data, aes(x = dominantCropGerberRayName, y = CATE_Value)) +
    geom_boxplot(outlier.shape = NA) +
    coord_cartesian(ylim = y_limits) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 14),
      axis.text.x = element_text(size = 14, angle = 45, hjust = 1),
      axis.title = element_text(size = 16),
      plot.title = element_text(size = 18, hjust = 0.5),
      legend.position = "none"
    ) +
    labs(title = title, x = "", y = "Change in yield gaps (%)")
}

p_erosion <- make_boxplot(filter(df_long, CATE_Type == "EROSION"), "Soil erosion debt", c(0, 0.2))
p_compaction <- make_boxplot(filter(df_long, CATE_Type == "COMPACTION"), "Soil compaction debt", c(0, 0.15))
p_soc <- make_boxplot(filter(df_long, CATE_Type == "SOC"), "Soil organic carbon debt", c(0, 0.05))
p_soilwater <- make_boxplot(filter(df_long, CATE_Type == "SOILWATER"), "Soil water debt", c(0, 0.075))
p_treecover <- make_boxplot(filter(df_long, CATE_Type == "TREECOVER"), "Tree cover debt", c(0, 0.15))
p_sum <- ggplot(df_sum_positive, aes(x = dominantCropGerberRayName, y = CATE_SUM)) +
  geom_boxplot(outlier.shape = NA) +
  coord_cartesian(ylim = c(0, 0.4)) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 14),
    axis.text.x = element_text(size = 14, angle = 45, hjust = 1),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 18, hjust = 0.5),
    legend.position = "none"
  ) +
  labs(title = "All land degradation", x = "", y = "Change in yield gaps (%)")

combined_plot <- plot_grid(
  p_erosion,
  p_compaction,
  p_soc,
  p_soilwater,
  p_treecover,
  p_sum,
  ncol = 2,
  align = "hv",
  labels = c("A", "B", "C", "D", "E", "F"),
  label_size = 16,
  label_x = 0,
  label_y = 1
)

ggsave(path_figure("SuppFig18_CateByCropType.pdf"), combined_plot, width = 18, height = 24)
ggsave(path_figure("SuppFig18_CateByCropType.png"), combined_plot, width = 18, height = 24, dpi = 300)
