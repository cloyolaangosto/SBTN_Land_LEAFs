# Generate Supplementary Fig. 18: CATE heterogeneity by dominant crop type.
#
# The figure summarizes saved post-fit CATE tables from models that include
# Ray/Gerber crop-type controls. 

library(dplyr)
library(ggplot2)
library(cowplot)
library(readr)
library(tibble)
library(tidyr)

source(file.path("code", "_shared", "paths.R"))

cf_dir <- path_processed("causal_forest")
linear_dataset_path <- path_processed("final_dataset", "analysis_ready_data_linear_scale.rds")
final_dataset_path <- path_processed("final_dataset", "analysis_ready_data_full.rds")
dataset_path <- if (file.exists(linear_dataset_path)) linear_dataset_path else final_dataset_path

cate_paths <- c(
  EROSION = file.path(cf_dir, "cate_erosion_crop_type_controls.rds"),
  COMPACTION = file.path(cf_dir, "cate_compaction_crop_type_controls.rds"),
  SOC = file.path(cf_dir, "cate_soc_crop_type_controls.rds"),
  SOILWATER = file.path(cf_dir, "cate_soil_water_crop_type_controls.rds"),
  TREECOVER = file.path(cf_dir, "cate_tree_cover_crop_type_controls.rds")
)

assert_files_exist(c(dataset_path, cate_paths))

crop_type_dummy_vars <- c(
  "dominantCropGerberRay_barley",
  "dominantCropGerberRay_cassava",
  "dominantCropGerberRay_maize",
  "dominantCropGerberRay_oilpalm",
  "dominantCropGerberRay_rapeseed",
  "dominantCropGerberRay_rice",
  "dominantCropGerberRay_sorghum",
  "dominantCropGerberRay_soybean",
  "dominantCropGerberRay_sugarcane",
  "dominantCropGerberRay_wheat"
)

read_cate <- function(process) {
  readRDS(cate_paths[[process]]) |>
    mutate(CATE = as.numeric(CATE)) |>
    rename(!!paste0("CATE_", process) := CATE)
}

df_ok_noNA <- readRDS(dataset_path) |>
  select(long, lat, all_of(crop_type_dummy_vars))

# Sum estimated elasticity coefficients across degradation indicators
# to obtain a combined association measure
df_CATE_crop_type_controls_merged <- read_cate("EROSION") |>
  inner_join(read_cate("COMPACTION"), by = c("long", "lat")) |>
  inner_join(read_cate("SOC"), by = c("long", "lat")) |>
  inner_join(read_cate("SOILWATER"), by = c("long", "lat")) |>
  inner_join(read_cate("TREECOVER"), by = c("long", "lat")) |>
  mutate(CATE_SUM = CATE_EROSION + CATE_COMPACTION + CATE_SOC + CATE_SOILWATER + CATE_TREECOVER) |>
  inner_join(df_ok_noNA, by = c("long", "lat")) |>
  mutate(
    dominantCropGerberRayName = case_when(
      dominantCropGerberRay_barley == 1 ~ "barley",
      dominantCropGerberRay_cassava == 1 ~ "cassava",
      dominantCropGerberRay_maize == 1 ~ "maize",
      dominantCropGerberRay_oilpalm == 1 ~ "oil palm",
      dominantCropGerberRay_rapeseed == 1 ~ "rapeseed",
      dominantCropGerberRay_rice == 1 ~ "rice",
      dominantCropGerberRay_sorghum == 1 ~ "sorghum",
      dominantCropGerberRay_soybean == 1 ~ "soybean",
      dominantCropGerberRay_sugarcane == 1 ~ "sugarcane",
      dominantCropGerberRay_wheat == 1 ~ "wheat",
      TRUE ~ "unknown"
    )
  )

df_long <- df_CATE_crop_type_controls_merged |>
  select(dominantCropGerberRayName, CATE_EROSION, CATE_COMPACTION, CATE_SOC, CATE_SOILWATER, CATE_TREECOVER) |>
  pivot_longer(
    cols = starts_with("CATE_"),
    names_to = "CATE_Type",
    values_to = "CATE_Value"
  ) |>
  mutate(CATE_Type = gsub("CATE_", "", CATE_Type)) |>
  filter(CATE_Value >= 0)

df_sum_positive <- df_CATE_crop_type_controls_merged |>
  filter(CATE_SUM >= 0)

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

saveRDS(
  list(
    data = df_sum_positive,
    process_data = df_long,
    input_files = c(dataset = dataset_path, cate_paths)
  ),
  path_figure_data("SuppFig18Data_CateByCropType.rds")
)
write_csv(df_sum_positive, path_figure_data("SuppFig18Data_CateByCropType.csv"))

