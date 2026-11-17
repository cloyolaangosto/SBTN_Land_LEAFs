# Compute production, calorie, protein, and revenue loss inputs.
#
# This stage combines FAOSTAT/FBS country information, harvested-area and yield
# rasters, and the total CATE raster to express the estimated yield-gap response
# as production, calorie, protein, and revenue losses at raster-cell and country
# scales.

library(dplyr)
library(readr)
library(tidyr)
library(tibble)

source(file.path("code", "_shared", "analysis_paths.R"))

loss_dir <- path_processed("losses")
cf_raster_dir <- path_processed("causal_forest", "rasterized")
raw_dir <- path_raw()
yield_gap_path <- path_raw("crop_yield", "yield_gap_2010_percent.tif")
country_gpkg_path <- path_raw("country_boundaries", "world_boundaries_fixed.gpkg")
fbs_csv_path <- path_raw("FoodBalanceSheets_E_All_Data_(Normalized)", "FoodBalanceSheets_E_All_Data_(Normalized).csv")
prices_csv_path <- path_raw("Prices_E_All_Data_(Normalized)", "Prices_E_All_Data_(Normalized).csv")
harvest_area_sum_path <- file.path(loss_dir, "harvested_area_selected_crops_ha.tif")
attain_yield_path <- file.path(raw_dir, "attainable_yields", "attainable_yield_2010_weighted.tif")
actual_yield_path <- file.path(raw_dir, "actual_yields", "actual_yield_2010_weighted.tif")

input_dependencies <- c(
  file.path(cf_raster_dir, "cate_sum_nonneg.tif"),
  yield_gap_path,
  country_gpkg_path,
  fbs_csv_path,
  prices_csv_path,
  attain_yield_path,
  actual_yield_path,
  harvest_area_sum_path
)


if (!allow_heavy("LDYG_ALLOW_LOSS_RECOMPUTE")) {
  stop(
    paste(
      "Loss raster/input computation can be slow; figure scripts use the saved outputs by default.",
      "Set LDYG_ALLOW_LOSS_RECOMPUTE=true to rebuild the loss rasters and country tables.",
      sep = "\n"
    ),
    call. = FALSE
  )
}

assert_files_exist(input_dependencies)

library(countrycode)
library(sf)
library(terra)

dir.create(loss_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# 2010 FAOSTAT/FBS aggregation for calorie, protein, and revenue conversion.
# ---------------------------------------------------------------------------

prepare_fao_fbs_inputs <- function(
  country_gpkg = country_gpkg_path,
  fbs_csv = fbs_csv_path,
  prices_csv = prices_csv_path
) {
  countries <- sf::st_read(country_gpkg, quiet = TRUE)
  if ("NAME" %in% names(countries)) {
    countries <- countries |> dplyr::filter(NAME != "Antarctica")
  }
  if ("iso3cd" %in% names(countries)) {
    countries <- countries |> rename(ISO3 = iso3cd)
  }
  if (!"ISO3" %in% names(countries) && "ISO3_CODE" %in% names(countries)) {
    countries <- countries |> rename(ISO3 = ISO3_CODE)
  }
  if (!"ISO3" %in% names(countries)) {
    stop("Country boundary file must contain ISO3, iso3cd, or ISO3_CODE.", call. = FALSE)
  }

  fbs <- readr::read_csv(fbs_csv, show_col_types = FALSE)
  prices <- readr::read_csv(prices_csv, show_col_types = FALSE)

  select_commodities_fbs <- c(
    "Wheat and products", "Rice and products", "Barley and products",
    "Maize and products", "Sorghum and products", "Cassava and products",
    "Sugar cane", "Rape and Mustardseed", "Soyabeans", "Palm Oil"
  )
  select_commodities_prices <- c(
    "Barley", "Maize (corn)", "Rice", "Sugar cane", "Wheat",
    "Sorghum", "Soya beans", "Rape or colza seed", "Cassava, fresh", "Palm oil"
  )
  item_match_table <- tribble(
    ~Item_in_prices, ~Item_in_fbs,
    "Barley", "Barley and products",
    "Maize (corn)", "Maize and products",
    "Rice", "Rice and products",
    "Soya beans", "Soyabeans",
    "Wheat", "Wheat and products",
    "Sorghum", "Sorghum and products",
    "Cassava, fresh", "Cassava and products",
    "Sugar cane", "Sugar cane",
    "Rape or colza seed", "Rape and Mustardseed",
    "Palm oil", "Palm Oil"
  )

  fbs_selCrop <- fbs |> filter(Item %in% select_commodities_fbs)
  prices_selCrop <- prices |> filter(Item %in% select_commodities_prices)

  fbs_selCrop_production <- fbs_selCrop |>
    filter(Element == "Production") |>
    select(`Area Code (M49)`, Area, Item, Year, Value) |>
    rename(production_tonne = Value) |>
    mutate(production_tonne = production_tonne * 1000)

  fbs_selCrop_food_supply <- fbs_selCrop |>
    filter(Element == "Food supply (kcal)") |>
    select(`Area Code (M49)`, Area, Item, Year, Value) |>
    rename(food_supply_kcal = Value) |>
    mutate(food_supply_kcal = food_supply_kcal * 1000000)
  fbs_selCrop_food <- fbs_selCrop |>
    filter(Element == "Food") |>
    select(`Area Code (M49)`, Area, Item, Year, Value) |>
    rename(food_quantity_tonne = Value) |>
    mutate(food_quantity_tonne = food_quantity_tonne * 1000)
  fbs_selCrop_protein_supply <- fbs_selCrop |>
    filter(Element == "Protein supply quantity (t)") |>
    select(`Area Code (M49)`, Area, Item, Year, Value) |>
    rename(protein_supply_tonne = Value)

  fbs_selCrop_food_all <- fbs_selCrop_food_supply |>
    left_join(fbs_selCrop_food, by = c("Area Code (M49)", "Area", "Item", "Year")) |>
    left_join(fbs_selCrop_protein_supply, by = c("Area Code (M49)", "Area", "Item", "Year")) |>
    select(`Area Code (M49)`, Area, Item, Year, food_supply_kcal, food_quantity_tonne, protein_supply_tonne) |>
    mutate(
      calories_kcal_per_tonne = food_supply_kcal / food_quantity_tonne,
      proteins_tonne_per_tonne = protein_supply_tonne / food_quantity_tonne
    )

  prices_selCrop_usd <- prices_selCrop |>
    filter(Element == "Producer Price (USD/tonne)") |>
    select(`Area Code (M49)`, Area, Item, Year, Value) |>
    rename(price_usd_per_tonne = Value)

  prices_selCrop_usd_add_production_tonne <- prices_selCrop_usd |>
    left_join(item_match_table, by = c("Item" = "Item_in_prices")) |>
    left_join(fbs_selCrop_production, by = c("Area Code (M49)", "Area", "Year", "Item_in_fbs" = "Item"))

  prices_selCrop_usd_selYear_aggYear <- prices_selCrop_usd |>
    filter(Year >= 2000) |>
    group_by(`Area Code (M49)`, Area, Item) |>
    mutate(diff = abs(Year - 2010)) |>
    filter(diff == min(diff)) |>
    ungroup() |>
    select(-diff)
  fbs_selCrop_production_selYear_aggYear <- fbs_selCrop_production |>
    group_by(`Area Code (M49)`, Area, Item) |>
    mutate(diff = abs(Year - 2010)) |>
    filter(diff == min(diff)) |>
    ungroup() |>
    select(-diff)
  prices_selCrop_usd_add_production_tonne_separateClosest2010 <- prices_selCrop_usd_selYear_aggYear |>
    left_join(item_match_table, by = c("Item" = "Item_in_prices")) |>
    left_join(fbs_selCrop_production_selYear_aggYear, by = c("Area Code (M49)", "Area", "Item_in_fbs" = "Item"))

  fbs_selCrop_food_all_selYear_aggYear <- fbs_selCrop_food_all |>
    group_by(`Area Code (M49)`, Area, Item) |>
    mutate(diff = abs(Year - 2010)) |>
    filter(diff == min(diff)) |>
    ungroup() |>
    select(-diff)
  prices_selCrop_usd_add_production_tonne_selYear_aggYear <- prices_selCrop_usd_add_production_tonne |>
    filter(Year > 2000) |>
    group_by(`Area Code (M49)`, Area, Item) |>
    mutate(diff = abs(Year - 2010)) |>
    filter(diff == min(diff)) |>
    ungroup() |>
    select(-diff)

  fbs_selCrop_food_all_selYear_aggYear_aggCrop <- fbs_selCrop_food_all_selYear_aggYear |>
    drop_na() |>
    filter(food_quantity_tonne > 0) |>
    group_by(`Area Code (M49)`, Area) |>
    summarise(across(c(food_supply_kcal, food_quantity_tonne, protein_supply_tonne), sum, na.rm = TRUE), .groups = "drop") |>
    mutate(
      calories_kcal_per_tonne = food_supply_kcal / food_quantity_tonne,
      proteins_tonne_per_tonne = protein_supply_tonne / food_quantity_tonne
    )
  fbs_selCrop_food_all_selYear_aggYear_aggCrop_TOTAL_CALORIES <- fbs_selCrop_food_all_selYear_aggYear |>
    drop_na() |>
    filter(food_quantity_tonne > 0) |>
    group_by(`Area Code (M49)`, Area) |>
    summarise(total_calories_kcal = sum(food_supply_kcal, na.rm = TRUE), .groups = "drop")
  fbs_selCrop_food_all_selYear_aggYear_aggCrop_TOTAL_PROTEINS <- fbs_selCrop_food_all_selYear_aggYear |>
    drop_na() |>
    filter(food_quantity_tonne > 0) |>
    group_by(`Area Code (M49)`, Area) |>
    summarise(total_proteins_tonne = sum(protein_supply_tonne, na.rm = TRUE), .groups = "drop")

  prices_selCrop_usd_add_production_tonne_selYear_aggYear_aggCrop <- prices_selCrop_usd_add_production_tonne_selYear_aggYear |>
    drop_na() |>
    filter(production_tonne > 0) |>
    group_by(`Area Code (M49)`, Area) |>
    summarise(price_usd_per_tonne = sum(price_usd_per_tonne * production_tonne, na.rm = TRUE) / sum(production_tonne, na.rm = TRUE), .groups = "drop")
  prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop <- prices_selCrop_usd_add_production_tonne_separateClosest2010 |>
    drop_na() |>
    filter(production_tonne > 0) |>
    group_by(`Area Code (M49)`, Area) |>
    summarise(price_usd_per_tonne = sum(price_usd_per_tonne * production_tonne, na.rm = TRUE) / sum(production_tonne, na.rm = TRUE), .groups = "drop")
  prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop_TOTAL_REVENUE <- prices_selCrop_usd_add_production_tonne_separateClosest2010 |>
    drop_na() |>
    filter(production_tonne > 0) |>
    group_by(`Area Code (M49)`, Area) |>
    summarise(total_revenues_usd = sum(price_usd_per_tonne * production_tonne, na.rm = TRUE), .groups = "drop")
  prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop_TOTAL_PRODUCTION <- prices_selCrop_usd_add_production_tonne_separateClosest2010 |>
    drop_na() |>
    filter(production_tonne > 0) |>
    group_by(`Area Code (M49)`, Area) |>
    summarise(total_production_tonne = sum(production_tonne, na.rm = TRUE), .groups = "drop")

  list(
    countries = countries,
    fbs_selCrop_food_all_selYear_aggYear_aggCrop = fbs_selCrop_food_all_selYear_aggYear_aggCrop,
    fbs_selCrop_food_all_selYear_aggYear_aggCrop_TOTAL_CALORIES = fbs_selCrop_food_all_selYear_aggYear_aggCrop_TOTAL_CALORIES,
    fbs_selCrop_food_all_selYear_aggYear_aggCrop_TOTAL_PROTEINS = fbs_selCrop_food_all_selYear_aggYear_aggCrop_TOTAL_PROTEINS,
    prices_selCrop_usd_add_production_tonne_selYear_aggYear_aggCrop = prices_selCrop_usd_add_production_tonne_selYear_aggYear_aggCrop,
    prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop = prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop,
    prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop_TOTAL_REVENUE = prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop_TOTAL_REVENUE,
    prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop_TOTAL_PRODUCTION = prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop_TOTAL_PRODUCTION
  )
}

save_country_fao_fbs_tables <- function(
  countries,
  fbs_selCrop_food_all_selYear_aggYear_aggCrop,
  fbs_selCrop_food_all_selYear_aggYear_aggCrop_TOTAL_CALORIES,
  fbs_selCrop_food_all_selYear_aggYear_aggCrop_TOTAL_PROTEINS,
  prices_selCrop_usd_add_production_tonne_selYear_aggYear_aggCrop,
  prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop,
  prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop_TOTAL_REVENUE,
  prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop_TOTAL_PRODUCTION
) {
  fbs_addISO <- fbs_selCrop_food_all_selYear_aggYear_aggCrop |>
    mutate(matched_iso3c = countrycode(as.numeric(sub("'", "", `Area Code (M49)`)), origin = "un", destination = "iso3c"))
  fbs_addISO_sf <- countries |>
    left_join(fbs_addISO, by = c("ISO3" = "matched_iso3c"))

  fbs_total_calories_sf <- fbs_selCrop_food_all_selYear_aggYear_aggCrop_TOTAL_CALORIES |>
    mutate(matched_iso3c = countrycode(as.numeric(sub("'", "", `Area Code (M49)`)), origin = "un", destination = "iso3c")) |>
    {\(x) countries |> left_join(x, by = c("ISO3" = "matched_iso3c"))}()
  write_rds(
    fbs_total_calories_sf,
    file.path(loss_dir, "fao_food_supply_calories_by_country.rds")
  )

  fbs_total_proteins_sf <- fbs_selCrop_food_all_selYear_aggYear_aggCrop_TOTAL_PROTEINS |>
    mutate(matched_iso3c = countrycode(as.numeric(sub("'", "", `Area Code (M49)`)), origin = "un", destination = "iso3c")) |>
    {\(x) countries |> left_join(x, by = c("ISO3" = "matched_iso3c"))}()
  write_rds(
    fbs_total_proteins_sf,
    file.path(loss_dir, "fao_food_supply_protein_by_country.rds")
  )

  prices_addISO_sf <- prices_selCrop_usd_add_production_tonne_selYear_aggYear_aggCrop |>
    mutate(matched_iso3c = countrycode(as.numeric(sub("'", "", `Area Code (M49)`)), origin = "un", destination = "iso3c")) |>
    {\(x) countries |> left_join(x, by = c("ISO3" = "matched_iso3c"))}()

  prices_separateClosest2010_sf <- prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop |>
    mutate(matched_iso3c = countrycode(as.numeric(sub("'", "", `Area Code (M49)`)), origin = "un", destination = "iso3c")) |>
    {\(x) countries |> left_join(x, by = c("ISO3" = "matched_iso3c"))}()

  prices_total_revenue_sf <- prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop_TOTAL_REVENUE |>
    mutate(matched_iso3c = countrycode(as.numeric(sub("'", "", `Area Code (M49)`)), origin = "un", destination = "iso3c")) |>
    {\(x) countries |> left_join(x, by = c("ISO3" = "matched_iso3c"))}()
  write_rds(
    prices_total_revenue_sf,
    file.path(loss_dir, "fao_revenue_by_country.rds")
  )

  prices_total_production_sf <- prices_selCrop_usd_add_production_tonne_separateClosest2010_aggCrop_TOTAL_PRODUCTION |>
    mutate(matched_iso3c = countrycode(as.numeric(sub("'", "", `Area Code (M49)`)), origin = "un", destination = "iso3c")) |>
    {\(x) countries |> left_join(x, by = c("ISO3" = "matched_iso3c"))}()
  write_rds(
    prices_total_production_sf,
    file.path(loss_dir, "fao_production_by_country.rds")
  )

  yieldGap_tr <- terra::rast(yield_gap_path)
  wopt_options <- list(
    gdal = c("COMPRESS=DEFLATE", "NUM_THREADS=10", "BLOCKXSIZE=512", "BLOCKYSIZE=512"),
    datatype = "FLT4S",
    NAflag = -3.4e+38
  )
  rasterize_func <- function(tr, field, filename = NULL) {
    terra::rasterize(
      x = tr,
      y = yieldGap_tr,
      field = field,
      fun = "max",
      filename = filename,
      overwrite = TRUE,
      wopt = wopt_options
    )
  }

  rasterize_func(
    fbs_addISO_sf,
    field = "calories_kcal_per_tonne",
    filename = file.path(loss_dir, "food_supply_calories_2010_kcal_per_tonne.tif")
  )
  rasterize_func(
    fbs_addISO_sf,
    field = "proteins_tonne_per_tonne",
    filename = file.path(loss_dir, "food_supply_protein_2010_tonne_per_tonne.tif")
  )
  rasterize_func(
    prices_addISO_sf,
    field = "price_usd_per_tonne",
    filename = file.path(loss_dir, "producer_price_2010_usd_per_tonne.tif")
  )
  rasterize_func(
    prices_separateClosest2010_sf,
    field = "price_usd_per_tonne",
    filename = file.path(loss_dir, "producer_price_2010_usd_per_tonne_closest_year.tif")
  )
}

# ---------------------------------------------------------------------------
# Raster algebra translating CATE estimates into production and nutrition losses.
# ---------------------------------------------------------------------------

compute_loss_rasters <- function(
  country_gpkg = country_gpkg_path,
  harvest_area_sum_path = harvest_area_sum_path,
  attain_yield_path = attain_yield_path,
  actual_yield_path = actual_yield_path
) {
  assert_files_exist(c(country_gpkg, attain_yield_path, actual_yield_path, harvest_area_sum_path))
  country_sf <- st_read(country_gpkg, quiet = TRUE) |>
    dplyr::filter(NAME != "Antarctica")
  country_tr <- vect(country_sf)

  cate_percent <- rast(file.path(cf_raster_dir, "cate_sum_nonneg.tif"))
  yield_gap <- rast(yield_gap_path)
  attainYieldAgg_tonne_per_ha <- rast(attain_yield_path)
  actualYieldAgg_tonne_per_ha <- rast(actual_yield_path)
  writeRaster(attainYieldAgg_tonne_per_ha, file.path(loss_dir, "attainable_yield_weighted_tonnes_per_ha.tif"), overwrite = TRUE)

  loss_yield_tonne_per_ha <- (((cate_percent / 100) * yield_gap) / 100) * attainYieldAgg_tonne_per_ha
  writeRaster(loss_yield_tonne_per_ha, file.path(loss_dir, "loss_yield_tonnes_per_ha.tif"), overwrite = TRUE)

  harvestArea_sum_ha <- rast(harvest_area_sum_path)

  loss_production_tonnes <- harvestArea_sum_ha * loss_yield_tonne_per_ha
  names(loss_production_tonnes) <- "loss_production_tonnes"
  writeRaster(loss_production_tonnes, file.path(loss_dir, "loss_production_tonnes.tif"), overwrite = TRUE)

  calories_million_kcal_per_tonne <- rast(file.path(loss_dir, "food_supply_calories_2010_kcal_per_tonne.tif")) / 1e6
  loss_calories_million_kcal <- loss_production_tonnes * calories_million_kcal_per_tonne
  names(loss_calories_million_kcal) <- "loss_calories_million_kcal"
  writeRaster(loss_calories_million_kcal, file.path(loss_dir, "loss_calories_million_kcal.tif"), overwrite = TRUE)

  proteins_tonne_per_tonne <- rast(file.path(loss_dir, "food_supply_protein_2010_tonne_per_tonne.tif"))
  loss_proteins_tonnes <- loss_production_tonnes * proteins_tonne_per_tonne
  names(loss_proteins_tonnes) <- "loss_proteins_tonnes"
  writeRaster(loss_proteins_tonnes, file.path(loss_dir, "loss_protein_tonnes.tif"), overwrite = TRUE)

  price_USD_per_tonne <- rast(file.path(loss_dir, "producer_price_2010_usd_per_tonne_closest_year.tif"))
  loss_revenues_USD <- loss_production_tonnes * price_USD_per_tonne
  names(loss_revenues_USD) <- "loss_revenues_USD"
  writeRaster(loss_revenues_USD, file.path(loss_dir, "loss_revenue_usd.tif"), overwrite = TRUE)

  actual_production_tonnes <- harvestArea_sum_ha * actualYieldAgg_tonne_per_ha
  actual_calories_million_kcal <- actual_production_tonnes * calories_million_kcal_per_tonne
  actual_proteins_tonnes <- actual_production_tonnes * proteins_tonne_per_tonne
  actual_revenues_USD <- actual_production_tonnes * price_USD_per_tonne

  extract_sum <- function(r, name) {
    terra::extract(r, country_tr, fun = sum, na.rm = TRUE) |>
      as_tibble() |>
      rename(!!name := 2)
  }

  country_loss_merged_sf <- country_sf |>
    bind_cols(extract_sum(loss_production_tonnes, "loss_production_tonnes") |> select(-ID)) |>
    bind_cols(extract_sum(loss_calories_million_kcal, "loss_calorie_million_kcal") |> select(-ID)) |>
    bind_cols(extract_sum(loss_proteins_tonnes, "loss_protein_tonnes") |> select(-ID)) |>
    bind_cols(extract_sum(loss_revenues_USD, "loss_revenue_usd") |> select(-ID)) |>
    bind_cols(extract_sum(actual_production_tonnes, "actual_production_tonnes") |> select(-ID)) |>
    bind_cols(extract_sum(actual_calories_million_kcal, "actual_calories_million_kcal") |> select(-ID)) |>
    bind_cols(extract_sum(actual_proteins_tonnes, "actual_proteins_tonnes") |> select(-ID)) |>
    bind_cols(extract_sum(actual_revenues_USD, "actual_revenues_USD") |> select(-ID)) |>
    mutate(
      total_production_tonnes = actual_production_tonnes,
      rel_loss_production_pct = 100 * loss_production_tonnes / actual_production_tonnes,
      rel_loss_calories_pct = 100 * loss_calorie_million_kcal / actual_calories_million_kcal,
      rel_loss_proteins_pct = 100 * loss_protein_tonnes / actual_proteins_tonnes,
      rel_loss_revenues_pct = 100 * loss_revenue_usd / actual_revenues_USD,
      rel_loss_pct = rel_loss_production_pct,
      rel_loss_frac = rel_loss_pct / 100
    )

  write_rds(country_loss_merged_sf, file.path(loss_dir, "country_loss_totals.rds"))
  st_write(country_loss_merged_sf, file.path(loss_dir, "country_loss_totals.gpkg"), delete_dsn = TRUE, quiet = TRUE)
}

fao_fbs_inputs <- prepare_fao_fbs_inputs()
do.call(save_country_fao_fbs_tables, fao_fbs_inputs)
compute_loss_rasters()



