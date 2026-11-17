# Create post-fit causal-forest inspection outputs for crop-type heterogeneity.
#
# This stage summarizes the Ray/Gerber crop-type-control CATE tables used for
# Supplementary Fig. 18. 


library(dplyr)
library(ggplot2)
library(cowplot)
library(readr)
library(tidyr)
library(tibble)

source(file.path("code", "_shared", "analysis_paths.R"))

cf_result_dir <- path_processed("causal_forest")
nf_rev_artifact_dir <- path_processed("crop_type_cate")
dir.create(nf_rev_artifact_dir, recursive = TRUE, showWarnings = FALSE)

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

input_dependencies <- c(
  file.path(cf_result_dir, "cate_erosion_crop_type_controls.rds"),
  file.path(cf_result_dir, "cate_compaction_crop_type_controls.rds"),
  file.path(cf_result_dir, "cate_soc_crop_type_controls.rds"),
  file.path(cf_result_dir, "cate_soil_water_crop_type_controls.rds"),
  file.path(cf_result_dir, "cate_tree_cover_crop_type_controls.rds"),
  path_processed("final_dataset", "analysis_ready_data_full.rds")
)


assert_files_exist(input_dependencies)

df_ok_noNA <- read_rds(path_processed("final_dataset", "analysis_ready_data_full.rds")) |>
  mutate(across(all_of(crop_type_dummy_vars), as.integer))

read_cate <- function(process) {
  clean_process <- tolower(process)
  clean_process <- gsub("soilwater", "soil_water", clean_process)
  clean_process <- gsub("treecover", "tree_cover", clean_process)
  read_rds(file.path(cf_result_dir, paste0("cate_", clean_process, "_crop_type_controls.rds"))) |>
    mutate(CATE = as.numeric(CATE)) |>
    rename(!!paste0("CATE_", process) := CATE)
}

# Sum estimated elasticity coefficients across degradation indicators
# to obtain a combined association measure
df_CATE_crop_type_controls_merged <- read_cate("EROSION") |>
  inner_join(read_cate("COMPACTION"), by = c("long", "lat")) |>
  inner_join(read_cate("SOC"), by = c("long", "lat")) |>
  inner_join(read_cate("SOILWATER"), by = c("long", "lat")) |>
  inner_join(read_cate("TREECOVER"), by = c("long", "lat")) |>
  mutate(CATE_SUM = CATE_EROSION + CATE_COMPACTION + CATE_SOC + CATE_SOILWATER + CATE_TREECOVER) |>
  inner_join(
    df_ok_noNA |> select(long, lat, all_of(crop_type_dummy_vars)),
    by = c("long", "lat")
  ) |>
  mutate(
    dominantCropGerberRayName = case_when(
      dominantCropGerberRay_barley == 1 ~ "Barley",
      dominantCropGerberRay_cassava == 1 ~ "Cassava",
      dominantCropGerberRay_maize == 1 ~ "Maize",
      dominantCropGerberRay_oilpalm == 1 ~ "Oil palm",
      dominantCropGerberRay_rapeseed == 1 ~ "Rapeseed",
      dominantCropGerberRay_rice == 1 ~ "Rice",
      dominantCropGerberRay_sorghum == 1 ~ "Sorghum",
      dominantCropGerberRay_soybean == 1 ~ "Soybean",
      dominantCropGerberRay_sugarcane == 1 ~ "Sugarcane",
      dominantCropGerberRay_wheat == 1 ~ "Wheat",
      TRUE ~ NA_character_
    )
  )

# For visualization and aggregation, retain only positive estimated CATEs to focus the analysis on
# locations where degradation is associated with higher yield gaps
df_sum_positive <- df_CATE_crop_type_controls_merged |>
  filter(CATE_SUM >= 0)

p_cate <- ggplot(df_sum_positive, aes(x = dominantCropGerberRayName, y = CATE_SUM)) +
  geom_boxplot(outlier.alpha = 0.05) +
  coord_flip() +
  theme_bw() +
  labs(x = NULL, y = "Summed CATE")

ggsave(file.path(nf_rev_artifact_dir, "SuppFig18_CateByCropType.pdf"), p_cate, width = 7, height = 5)
ggsave(file.path(nf_rev_artifact_dir, "SuppFig18_CateByCropType.png"), p_cate, width = 7, height = 5, dpi = 300)



