# Create figure data used by the CATE map figures.

source(file.path("code", "_shared", "cate_map_helpers.R"))

assert_cate_base_inputs(include_yield_gap = TRUE)
assert_files_exist(tuned_inputs)

admin <- read_cate_admin()
cate_sum <- read_cate_sum()
yield_gap <- read_yield_gap()

individual_stats <- lapply(names(individual_paths), function(id) {
  quantile_tbl(rast(individual_paths[[id]]), id)
})

figure_stats <- bind_rows(
  quantile_tbl(cate_sum, "CATE_sum"),
  bind_rows(individual_stats)
)

write_raster_figure_data <- function(basename, rasters, input_files) {
  figure_data <- list(
    rasters = lapply(rasters, terra::wrap),
    admin = admin,
    input_files = input_files,
    quantiles = figure_stats,
    map_maxcell = Sys.getenv("LDYG_MAP_MAXCELL", unset = "200000")
  )

  saveRDS(figure_data, path_figure_data(paste0(basename, ".rds")))
  write_csv(figure_stats, path_figure_data(paste0(basename, "RasterQuantiles.csv")))
}

write_raster_figure_data(
  "Fig4Data_MapOverallLandDegradationCate",
  list(cate_sum = cate_sum, yield_gap = yield_gap),
  c(cate_sum = cate_sum_path, yield_gap = yield_gap_path)
)

write_raster_figure_data(
  "SuppFig7Data_MapOverallLandDegradationCateFisher7",
  list(cate_sum = cate_sum),
  c(cate_sum = cate_sum_path)
)

individual_data_basenames <- c(
  EROSION = "SuppFig8Data_MapCateErosion",
  COMPACTION = "SuppFig9Data_MapCateCompaction",
  SOC = "SuppFig10Data_MapCateSoc",
  SOILWATER = "SuppFig11Data_MapCateSoilWater",
  TREECOVER = "SuppFig12Data_MapCateTreeCover"
)

for (id in names(individual_data_basenames)) {
  spec <- individual_specs[[id]]
  r <- rast(spec$path)
  names(r) <- "CATE"
  write_raster_figure_data(
    individual_data_basenames[[id]],
    list(cate = r),
    c(cate = spec$path)
  )
}

tuned_data_basenames <- c(
  SuppFig13_MapCateErosionTuned = "SuppFig13Data_MapCateErosionTuned",
  SuppFig14_MapCateErosionSeparateModels = "SuppFig14Data_MapCateErosionSeparateModels"
)

for (id in names(tuned_data_basenames)) {
  df_cate <- readRDS(tuned_inputs[[id]])
  figure_data <- list(
    data = df_cate,
    yield_gap = terra::wrap(yield_gap),
    admin = admin,
    input_files = c(data = tuned_inputs[[id]], yield_gap = yield_gap_path),
    quantiles = figure_stats,
    map_maxcell = Sys.getenv("LDYG_MAP_MAXCELL", unset = "200000")
  )

  basename <- tuned_data_basenames[[id]]
  saveRDS(figure_data, path_figure_data(paste0(basename, ".rds")))
  write_csv(figure_stats, path_figure_data(paste0(basename, "RasterQuantiles.csv")))
}
