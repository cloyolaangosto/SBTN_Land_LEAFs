# Create figure data used by the production, nutrition, and revenue loss figures.

source(file.path("code", "_shared", "loss_figure_helpers.R"))

assert_loss_inputs()

admin <- read_loss_admin()
country_abs <- read_country_loss()

country_loss_export <- country_abs |>
  st_drop_geometry() |>
  select(
    ISO3,
    NAME,
    total_production_tonnes,
    loss_production_tonnes,
    loss_calorie_million_kcal,
    loss_revenue_usd,
    loss_protein_tonnes,
    rel_loss_pct,
    countrySum_loss_production_tonnes_thousand,
    countrySum_loss_calories_billion_kcal,
    countrySum_loss_revenues_million_USD,
    countrySum_loss_proteins_tonnes
  )

top_bar_data <- bind_rows(
  country_loss_export |>
    filter(!is.na(countrySum_loss_production_tonnes_thousand)) |>
    slice_max(countrySum_loss_production_tonnes_thousand, n = 7) |>
    transmute(panel = "E", metric = "production", ISO3, NAME, value = countrySum_loss_production_tonnes_thousand),
  country_loss_export |>
    filter(!is.na(countrySum_loss_calories_billion_kcal)) |>
    slice_max(countrySum_loss_calories_billion_kcal, n = 7) |>
    transmute(panel = "F", metric = "calories", ISO3, NAME, value = countrySum_loss_calories_billion_kcal),
  country_loss_export |>
    filter(!is.na(countrySum_loss_revenues_million_USD)) |>
    slice_max(countrySum_loss_revenues_million_USD, n = 7) |>
    transmute(panel = "G", metric = "revenues", ISO3, NAME, value = countrySum_loss_revenues_million_USD)
)

relative_loss_top_bar_data <- country_loss_export |>
  filter(!is.na(rel_loss_pct)) |>
  slice_max(rel_loss_pct, n = 7) |>
  transmute(panel = "B", metric = "relative_production_loss", ISO3, NAME, value = rel_loss_pct)

raster_quantiles <- bind_rows(
  quantile_tbl(rast(loss_paths[["production"]]), "loss_production_tonnes"),
  quantile_tbl(rast(loss_paths[["calories"]]), "loss_calories_million_kcal"),
  quantile_tbl(rast(loss_paths[["proteins"]]), "loss_proteins_tonnes"),
  quantile_tbl(rast(loss_paths[["revenues"]]), "loss_revenues_USD"),
  quantile_tbl(rast(input_paths[["harvest_area"]]), "harvest_area_ha"),
  quantile_tbl(rast(input_paths[["attainable_yield"]]), "attainable_yield_tonnes_per_ha"),
  quantile_tbl(rast(input_paths[["price"]]), "producer_price_usd_per_tonne"),
  quantile_tbl(rast(input_paths[["calories"]]), "calories_kcal_per_tonne"),
  quantile_tbl(rast(input_paths[["proteins"]]), "proteins_tonne_per_tonne")
)

loss_rasters <- lapply(loss_paths, function(path) terra::wrap(rast(path)))
input_rasters <- lapply(input_paths, function(path) terra::wrap(rast(path)))

base_figure_data <- list(
  country_loss = country_loss_export,
  country_loss_sf = country_abs,
  admin = admin,
  loss_rasters = loss_rasters,
  input_rasters = input_rasters,
  raster_quantiles = raster_quantiles,
  input_files = c(country_loss = country_loss_path, loss_paths, input_paths),
  map_maxcell = Sys.getenv("LDYG_MAP_MAXCELL", unset = "200000")
)

fig5_data <- c(
  base_figure_data,
  list(
    map_panels_A_D = country_loss_export,
    barplot_panels_E_G = top_bar_data
  )
)

fig6_data <- c(
  base_figure_data,
  list(
    map_panel_A = country_loss_export,
    barplot_panel_B = relative_loss_top_bar_data
  )
)

write_csv(country_loss_export, path_figure_data("Fig5Data_MapCountryLosses.csv"))
write_csv(top_bar_data, path_figure_data("Fig5Data_BarplotCountryLosses.csv"))
write_csv(country_loss_export, path_figure_data("SuppFig16Data_MapCountryLossesContinuous.csv"))
write_csv(country_loss_export, path_figure_data("Fig6Data_MapRelativeProductionLosses.csv"))

saveRDS(fig5_data, path_figure_data("Fig5Data_CountryLosses.rds"))
saveRDS(fig5_data, path_figure_data("Fig5Data_MapCountryLosses.rds"))
saveRDS(fig5_data, path_figure_data("Fig5Data_BarplotCountryLosses.rds"))
saveRDS(fig6_data, path_figure_data("Fig6Data_MapRelativeProductionLosses.rds"))
saveRDS(base_figure_data, path_figure_data("SuppFig15Data_MapPixelLevelLosses.rds"))
saveRDS(base_figure_data, path_figure_data("SuppFig16Data_MapCountryLossesContinuous.rds"))
saveRDS(base_figure_data, path_figure_data("SuppFig17Data_MapInputDataLossCalculations.rds"))

write_csv(raster_quantiles, path_figure_data("SuppFig15Data_MapPixelLevelLossesRasterQuantiles.csv"))
write_csv(raster_quantiles, path_figure_data("SuppFig17Data_MapInputDataLossCalculationsRasterQuantiles.csv"))
