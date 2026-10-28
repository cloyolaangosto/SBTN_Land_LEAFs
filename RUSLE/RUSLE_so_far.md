# RUSLE so far — current state of the Soil Erosion LEAFs

Summary of every piece of RUSLE / soil-erosion documentation and code currently in the repository (branch `RUSLE_Claude_proposal1`), the data it uses, and the known gaps that motivate the new R, K and LS layers described in [README.md](README.md).

Sources reviewed:

- [documentation/Soil_Erosion_Documentation.md](../documentation/Soil_Erosion_Documentation.md) — method description
- [documentation/SoilErosion_Baseline_LEAFs.ipynb](../documentation/SoilErosion_Baseline_LEAFs.ipynb) — baseline LEAF generation (code)
- [documentation/SoilErosion_LEAFs_ReducedTillage.ipynb](../documentation/SoilErosion_LEAFs_ReducedTillage.ipynb) — reduced-tillage LEAFs (code)
- [documentation/Challenges_and_Learnings.md](../documentation/Challenges_and_Learnings.md), [documentation/LEAFs_Overview_NonExpert.md](../documentation/LEAFs_Overview_NonExpert.md), [documentation/SOC_Documentation.md](../documentation/SOC_Documentation.md) — context, shared data sources
- [LEAFs/README.md](../LEAFs/README.md), [LEAFs/soil_erosion/aggregation_analysis/README.md](../LEAFs/soil_erosion/aggregation_analysis/README.md) — outputs and aggregation findings
- [src/sbtn_leaf/map_calculations.py](../src/sbtn_leaf/map_calculations.py) — raster helpers used by the notebooks
- Contents of `data/soil_erosion/` and `data/soil_weather/`

---

## 1. Model

Soil erosion LEAFs (t soil ha⁻¹ yr⁻¹) are computed with the Revised Universal Soil Loss Equation:

$$A = R \cdot K_{st} \cdot LS \cdot C \cdot P$$

| Factor | Meaning | Unit | Current source |
|---|---|---|---|
| $R$ | Rainfall-runoff erosivity | MJ mm ha⁻¹ h⁻¹ yr⁻¹ | GloSEM v1.1 (25 km) |
| $K_{st}$ | Soil erodibility, corrected for stone (skeletal) fraction | t ha h ha⁻¹ MJ⁻¹ mm⁻¹ | GloSEM v1.1 (25 km) |
| $LS$ | Slope length and steepness | – | GloSEM v1.1 (25 km) |
| $C$ | Cover and management | – | Borrelli et al. (2017) crop/land-cover tables + Panagos et al. (2015) management sub-factors |
| $P$ | Support practices | – | Assumed 1 (common in LCA characterisation factors) |

Design choice: $R$, $K_{st}$ and $LS$ do not depend on land use, so they are multiplied once into a single **RLSK base layer**. Each land use and management combination then only needs its own $C$ raster.

## 2. Data used

### 2.1 Soil and weather (R, K, LS)

- **GloSEM v1.1, 25 km release** (ESDAC, [download](https://esdac.jrc.ec.europa.eu/public_path/shared_folder/dataset/58/GloSEM_25km.zip)), from Borrelli et al. (2017) and Panagos et al. (2017):
  - `data/soil_erosion/GloSEM_v1.1/RUSLE_RFactor_v1.1_25km.tif`
  - `data/soil_erosion/GloSEM_v1.1/RUSLE_KFactor_v1.1_25km.tif`
  - `data/soil_erosion/GloSEM_v1.1/RUSLE_LSFactor_v1.1_25km.tif`
  - Also downloaded but not used: `RUSLE_CFactor_yr2012_v1.1_25km.tif`, `RUSLE_SoilLoss_v1.1_yr2001_25km.tif`, `RUSLE_SoilLoss_v1.1_yr2012_25km.tif`.
- Grid (checked with rasterio): EPSG:4326, cell size 0.208333° (12.5 arc-min, ≈23 km at the equator), 1728 × 673 cells, latitude −56.6° to 83.6°. R is float64; K and LS are float32.
- Product: `data/soil_erosion/rlsk_2012.tif` = R × LS × K, made with `mc.multiply_rasters`. The "2012" in the name refers to the GloSEM reference year, not to a year-specific R.

### 2.2 Cover management (C)

- **Land-use maps:** UHTH zones of Morais, Teixeira & Domingos (2019), built from FAO GAEZ v2 suitability. These are the same land-use maps used for the SOC LEAFs, on a **5 arc-min grid** (0.083333°, ≈9.3 km; 4320 × 2083 cells; latitude −90° to 83.6°).
- **Crop and land-cover C values:** Borrelli et al. (2017), Tables 1–2 of the soil erosion documentation (e.g. cereals 0.20, roots and tubers 0.34, tobacco 0.50, forests 0.0001–0.003, grassland 0.01–0.15). They are mapped to each land use in `data/crops/lu_c_factor_inventory.xlsx`.
- **Management sub-factors**, after Panagos et al. (2015):
  - $C_{arable} = C_{crop} \cdot C_{mgmt}$
  - $C_{mgmt} = C_{tillage} \cdot C_{residues} \cdot C_{cover}$
  - Tillage: 1 (conventional), 0.35 (reduced/conservation), 0.25 (no-till)
  - $C_{residues} = 1 - 0.12\,F_{residues}$
  - $C_{cover} = 1 - 0.2\,F_{cover}$
  - Reduced-tillage combinations are listed in `data/crops/lu_c_factor_reducedtillage.xlsx`.
- **Rasters:**
  - `data/soil_erosion/C_Factors/10km/cfactor_10km_*.tif`, on the 5′ UHTH grid, made with `mc.create_binary_mask`.
  - `data/soil_erosion/C_Factors/25km/cfactor_25km_*.tif`, resampled to the RLSK grid with `mc.resample_to_match` (nearest neighbour).

### 2.3 Other soil and weather layers already on disk (5′ UHTH grid, used by SOC)

| File | Content | Original source |
|---|---|---|
| `data/soil_weather/uhth_monthly_avg_precip.tif` | 12-band mean monthly precipitation (mm/month), 2013–2023 average | NASA GPM IMERG (GPM_3IMERGM v07, 0.1°) |
| `data/soil_weather/uhth_monthly_avg_temp_celsius.tif` | 12-band mean monthly temperature | NASA GLDAS CLSM (1°) |
| `data/soil_weather/uhth_pet_locationonly.tif` | 12-band PET | – |
| `data/soil_weather/uhth_sand_15-30cm_mean_perc.tif` | Sand % (15–30 cm) | ISRIC SoilGrids 250 m |
| `data/soil_weather/uhth_clay_15-30cm_mean_perc.tif` | Clay % (15–30 cm) | ISRIC SoilGrids 250 m |
| `data/soil_weather/uhth_soc_0-30cm_mean.tif` | SOC stock 0–30 cm (t C/ha) | ISRIC SoilGrids 250 m |
| `data/soil_weather/uhth_thermal_climates.tif` | Thermal climate zones | GAEZ |

These layers already sit on the target 5′ grid and are relevant to the new K (texture, SOC) and R (monthly precipitation) layers. See README §"Data already available".

## 3. Processing pipeline (as implemented in the notebooks)

1. **RLSK layer:** multiply the GloSEM v1.1 R, LS and K rasters. They share one grid, so no harmonisation is needed. The result is `rlsk_2012.tif`.
2. **C rasters:** for each row in `lu_c_factor_inventory.xlsx`, burn `c_final` into the land-use footprint at 5′, then resample to the 25 km RLSK grid with nearest neighbour.
3. **Erosion rate:** `se_rate_25km_<land use>.tif = RLSK × C`, written to `LEAFs/soil_erosion/raster/`.
4. **Outlier treatment:** pixels above 1,000 t ha⁻¹ yr⁻¹ were found. Rasters are clipped with `filter_raster_outliers`:
   - log1p winsorisation at the 0.5–99.5 percentiles;
   - local z-score filter (3×3 window, k = 2.5, at least 8 neighbours);
   - global cap at the 99.5th percentile.
   - Results go to `LEAFs/soil_erosion/rasters_clipped/`.
5. **Regional LEAFs:** area-weighted mean, median and SD per country (GAUL level 0, dissolved), subcountry (GAUL level 1, `ADM1_CODE`) and ecoregion (Ecoregions2017, `ECO_ID`).
   - Uses `mc.build_cfs_gpkg_from_rasters` with `outlier_method="log1p_win"`: q = 0.005–0.995 for countries, 0.01–0.99 for subcountries and ecoregions.
   - Outputs: `se_country_clipped_2`, `se_subcountry_clipped_2`, `se_ecoregions_clipped_2` (.gpkg and .csv).
6. **Post-processing:** flow names harmonised via `paper/se_flows.csv`; Tundra and Antarctica ecoregions removed; columns renamed to `leaf`, `leaf_median`, `leaf_std`.
7. **Reduced tillage:** same pipeline with $C_{tillage} = 0.35$ for cereals and oilseeds (`*_rt` flows).

Outputs: **106 flows** (crops × residue × tillage, plus forests, grassland and urban) across 42 land-use classes. Coverage is 51% for countries, 58% for subcountries and 65% for ecoregions. Within-country variance is substantial: 23–46% of admin-1 variance is hidden inside a national LEAF (see `LEAFs/soil_erosion/aggregation_analysis/README.md`).

## 4. Known gaps and issues

| # | Issue | Consequence |
|---|---|---|
| 1 | R, K and LS come from **GloSEM v1.1 at 25 km**, while C is native 5′ (~10 km). C is down-sampled with **nearest neighbour**. | Most of the C detail is thrown away. Steep or erosive pixels are averaged with flat ones inside 25 km cells, which smooths the LS signal heavily. This is the main reason for this branch. |
| 2 | GloSEM v1.3 (Borrelli et al., 2022) releases only **final soil-displacement rasters (100 m, cropland only)**, not separate R/K/LS layers. | The base layers cannot simply be "upgraded to 1.3". R, K and LS must be rebuilt or sourced separately (see README). |
| 3 | R is a long-term mean (single static layer). | No climate scenarios, interannual variability or hazard (drought/flood) effects are possible yet. |
| 4 | $P = 1$ everywhere. | Terracing and contouring are not represented (acceptable for LCA-type factors). |
| 5 | Documentation TODOs: missing RUSLE and P=1 citations; UHTH resolution "TODO"; raster download location "XXX"; in Soil_Erosion_Documentation.md the "Borrelli et al., 2022" link points to the 2017 paper. | Fix while updating the documentation. |
| 6 | The documentation says "25 km" but the grid is 0.2083° (≈23 km at the equator). | Cosmetic; state the exact cell size in new docs. |
| 7 | `data/soil_erosion/sf_2012.tif` (same size as `rlsk_2012.tif`) is not referenced in any notebook or doc. | Identify or remove. |
| 8 | `mc.resample_to_match` defaults to `Resampling.nearest` and reads whole bands into memory. | For aggregating 1 km / 90 m inputs to 5′, pass `Resampling.average`, and read windowed or tiled for global 90 m data. |
