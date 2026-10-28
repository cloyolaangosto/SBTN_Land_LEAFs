# Purpose
This branch creates new RUSLE LEAFs at higher resolution, using publications from 2024 onwards. The original calculations use GloSEM 1.1 at a pixel resolution of 25 km. The goal is new R, K and LS layers at 10 km or finer, since the C and P layers are already at higher resolution.

The R layer depends on rainfall and changes over time. K depends on soil properties and LS on terrain, so neither changes much over time and both can be treated as static.

Related documents in this folder:
- [RUSLE_so_far.md](RUSLE_so_far.md) — current pipeline, data and known gaps of the existing soil-erosion LEAFs.
- [papers/papers_summary.md](papers/papers_summary.md) — summary of the three RUSLE method papers.
- [papers_risk/papers_risk_summary.md](papers_risk/papers_risk_summary.md) — summary of the climate-risk and hazard papers used for the stochastic R design.

# New Layers
Three new layers are needed: R, K and LS. ("L" in earlier notes refers to the combined slope length and steepness factor, LS.) Three papers are being explored for this:
1. [Borrelli et al. (2022). GloSEM: High-resolution global estimates of present and future soil displacement in croplands by water erosion](https://www.nature.com/articles/s41597-022-01489-x)
2. [Fadl et al. (2025). Integrating RUSLE, AHP, GIS, and cloud-based geospatial analysis for soil erosion assessment under mediterranean conditions](https://www.nature.com/articles/s41598-025-22503-3)
3. [Pyne et al. (2026). Analyzing soil erosion trends and future predictions using the RUSLE model: A case study of the Dwarakeswar River Basin, West Bengal](https://www.sciencedirect.com/science/article/pii/S2095633925000930)

The choice of method weighs data accessibility, what has already been downloaded, and ease of implementation. Borrelli et al. (2022) gets priority, because an earlier version of the GloSEM model was used (1.1 instead of the current 1.3).

> **Key constraint.** GloSEM 1.3 publishes only the *final* soil-displacement rasters (100 m, cropland only). It does not publish separate R, K or LS layers. Our LEAF scheme needs those base factors separately, so that C can vary by crop and management. Therefore:
> - **R** comes from the published GloREDa / GloSEM-lineage erosivity maps (same GPR method as GloSEM).
> - **K** and **LS** are rebuilt with the GloSEM equations, on open soil and DEM data.

## Target grid
All new layers are produced on the **UHTH 5 arc-min grid** (0.083333°, ≈9.3 km at the equator; EPSG:4326; 4320 × 2083 cells; latitude −90° to 83.58°). This is the grid of the existing C-factor rasters (`data/soil_erosion/C_Factors/10km/`) and of all SOC inputs (`data/soil_weather/`). Using it means:
- C no longer has to be down-sampled to 25 km with nearest neighbour.
- Soil erosion and SOC LEAFs share one grid.
- Any existing `uhth_*.tif` raster can serve as the template for `mc.resample_raster_to_match`.

Fine-resolution inputs (1 km, 250 m, 90 m) are **aggregated by mean** (`Resampling.average`, or windowed or tiled processing for 90 m data), not by nearest neighbour. For non-linear factors (K, LS), the factor is computed at native resolution *before* aggregating.

## Data already available
| Data on disk | Location | Grid | Usable for |
|---|---|---|---|
| GloSEM v1.1 R, K, LS (25 km) | `data/soil_erosion/GloSEM_v1.1/` | 0.2083° | **Validation** of the new layers only |
| Mean monthly precipitation, IMERG 2013–2023 (12 bands) | `data/soil_weather/uhth_monthly_avg_precip.tif` | 5′ | R: Modified Fournier Index (MFI) baseline / sanity check |
| Sand % and clay % (15–30 cm), SoilGrids | `data/soil_weather/uhth_{sand,clay}_15-30cm_mean_perc.tif` | 5′ | K: preliminary run only. Wrong depth (topsoil 0–30 cm preferred), and already aggregated to 5′ |
| SOC stock 0–30 cm (t C/ha), SoilGrids | `data/soil_weather/uhth_soc_0-30cm_mean.tif` | 5′ | K: only via a bulk-density conversion to SOC %. Better to download SOC concentration |
| Mean monthly temperature, PET | `data/soil_weather/uhth_*` | 5′ | Hazard: SPEI-type drought index (precipitation − PET) |
| C-factor rasters, all LEAF flows | `data/soil_erosion/C_Factors/10km/` | 5′ | Multiply directly with the new RLSK |

Nothing usable has been downloaded yet for **LS** (no DEM or flow accumulation), **future precipitation**, or **interannual and extreme precipitation**.

## R Layer (Rainfall erosivity)
The R layer depends on rainfall. It should be adjustable for different future precipitation scenarios (IPCC SSPs), and should be able to estimate the extra erosion from hazards (flood/extreme rain, drought). It must stay flexible enough to run different scenarios.

### Methods
| # | Method | Paper | Formula / approach | Inputs | Scenario-ready | Effort |
|---|---|---|---|---|---|---|
| R1 | **GPR on station erosivity** (GloREDa) | Borrelli et al. 2022 (via Panagos et al. 2017, 2022, 2023) | $R=\frac{1}{n}\sum_j\sum_k (EI_{30})_k$ at 3,625–3,939 stations, interpolated with Gaussian Process Regression (or ensemble ML for monthly) on WorldClim covariates | Station data (not public). **Outputs are public**: annual R 1 km; monthly R 1 km (12 GeoTIFFs); future R 1 km for 2050/2070 × RCP2.6/4.5/8.5 (CMIP5, 19 GCMs) | Yes (CMIP5 maps) | Re-fitting: very high. **Using the published maps: low** |
| R2 | **Modified Fournier Index** (Arnoldus 1980) | Pyne et al. 2026 | $R=\sum_{i=1}^{12}1.735\cdot10^{\,1.5\log_{10}(P_i^2/P)-0.08188}$ (US units; × 17.02 for SI) | Monthly precipitation only | **Yes**, any monthly precipitation scenario | Low |
| R3 | **Linear regression on annual P** | Fadl et al. 2025 | $R = 79 + 0.363\,P_{annual}$ (regional Indian coefficients; equation garbled in the paper) | Annual precipitation | Weakly; ignores seasonality and intensity | Very low, but not transferable globally |

### Selected R method
**Hybrid R1 + R2:**
- **Baseline magnitude and spatial pattern**: the published GloREDa R map, same lineage as GloSEM.
- **Scenario signal**: a **ratio (delta) of Modified Fournier Indices** computed from monthly precipitation of any scenario.
- **Stochastic module**: interannual variability, drought and extreme rain.

Why:
- The station GPR cannot be re-fitted (no station data), but its 1 km output exists and keeps continuity with GloSEM.
- The MFI ratio needs only monthly precipitation, which every CMIP6 product supplies. Its regional calibration constants and US-unit factor cancel in the ratio.
- The published CMIP5 future-R maps (Panagos et al. 2022) act as an independent cross-check of the scenario ratios.

Details are in [Incorporating precipitation changes into R layer](#incorporating-precipitation-changes-into-r-layer) and [Incorporating Risk Hazards](#incorporating-risk-hazards).

## K and LS layers
These layers are constant. They can be calculated once and stored as GeoTIFFs, but they still need to move to higher resolution.

### Methods — K (soil erodibility)
| # | Method | Paper | Formula | Inputs | Effort |
|---|---|---|---|---|---|
| K1 | **Wischmeier & Smith (1978) nomograph** + stone correction | Borrelli et al. 2022 (GloSEM; also current v1.1 layer) | $K=\frac{2.1\cdot10^{-4}M^{1.14}(12-OM)+3.25(s-2)+2.5(p-3)}{100}\cdot0.1317$, with $M=(\%silt+\%vfs)(100-\%clay)$; $K_{st}=K\cdot e^{-0.04(R_c-10)}$ for $R_c>10\%$ | Sand, silt, clay, OM, coarse fragments (SoilGrids); structure class $s$ and permeability class $p$ from texture-class lookups; very fine sand ($vfs$) estimated (not in SoilGrids) | Medium (lookups and a vfs assumption) |
| K2 | **Williams / EPIC** (Sharpley & Williams 1990) | Pyne et al. 2026 | $K=[0.2+0.3e^{-0.0256Sd(1-St/100)}]\,[\frac{St}{Cy+St}]^{0.3}\,[1-\frac{0.25OC}{OC+e^{3.72-2.95OC}}]\,[1-\frac{0.7SN}{SN+e^{-5.51+22.9SN}}]\times0.1317$ | Sand, silt, clay, OC (SoilGrids) | **Low**, pure raster algebra |
| K3 | **Renard et al. (1997) geometric mean diameter** | Fadl et al. 2025 | $K=0.0034+0.0405\exp[-0.5(\frac{\log D_g+1.659}{0.7101})^2]$ | Sand, silt, clay | Very low, but ignores OM and stones |

### Methods — LS (slope length and steepness)
| # | Method | Paper | Formula / approach | Inputs | Effort |
|---|---|---|---|---|---|
| LS1 | **Desmet & Govers (1996)** 2-D unit contributing area | Borrelli et al. 2022 | $L_{i,j}=\frac{(A_{i,j}+D^2)^{m+1}-A_{i,j}^{m+1}}{D^{m+2}\,x_{i,j}^m\,22.13^m}$ with $m=\beta/(1+\beta)$, $\beta=\frac{\sin\theta/0.0896}{3\sin^{0.8}\theta+0.56}$; S from McCool (1987) | 90 m DEM and upslope contributing area (SRTM/ASTER in GloSEM; SAGA, 870 tiles) | High if flow routing is done in-house. **Medium if precomputed flow accumulation is used** (MERIT Hydro) |
| LS2 | **Slope-class exponent + McCool S** (Liu steep-slope variant) | Pyne et al. 2026 | $L=(\lambda/22.13)^m$, $m$ = 0.2/0.3/0.4/0.5 by slope class; S = 10.8 sinθ + 0.03 (< 9%), 16.8 sinθ − 0.5 (9–18%), 21.9 sinθ − 0.96 (≥ 18%) | DEM, flow length | Medium |
| LS3 | **Shinde et al. (2011)** | Fadl et al. 2025 | $LS=(L/22.1)^m(65.4\sin^2\theta+4.56\sin\theta+0.0065)$ | DEM, slope length | Low, but uncapped lengths give unrealistic LS (up to 4,110 in the paper) |

### Selected K and LS methods
- **K → K1 (GloSEM nomograph + stone correction), with K2 (EPIC) computed alongside as a check.**
  - K1 is what GloSEM 1.1/1.3 and the current $K_{st}$ layer use, so the new layer stays comparable with the existing LEAFs.
  - K2 is almost free once the SoilGrids layers are downloaded. Where the two disagree strongly, the nomograph assumptions (vfs, $s$, $p$) can be flagged.
  - If the vfs and structure lookups cannot be reproduced defensibly, **fall back to K2**. It is the easiest method fully supported by open data.
- **LS → LS1 (Desmet & Govers, as GloSEM), using precomputed MERIT Hydro flow accumulation and MERIT DEM slope at 3″ (~90 m), aggregated to 5′.**
  - This reproduces the GloSEM method without running global flow routing.
  - It can run in Google Earth Engine, which hosts MERIT DEM and MERIT Hydro, or locally tile by tile.

## Selected method

### R layer
**Baseline:** $R_{base}$ is the GloREDa global annual R map (Panagos et al. 2017; 30″ ≈ 1 km; ESDAC). Optionally, the sum of the 12 monthly GloREDa rasters (Panagos et al. 2023) can be used instead. That version also gives the **monthly distribution of erosivity**, which is useful later for matching erosivity to crop calendars or seasonal C. $R_{base}$ is aggregated to the 5′ grid by mean.

**Scenario:** $R_s = R_{base}\cdot\Delta_s$, with $\Delta_s$ the MFI ratio described in the next section.

**Stochastic:** $R_{y,s} = R_{base}\cdot\Delta_s\cdot\varepsilon_y\cdot h_y$; see [Incorporating Risk Hazards](#incorporating-risk-hazards).

Output files:
- `rusle_R_5m_baseline.tif`
- `rusle_R_5m_<ssp>_<period>_<gcm|ensemble>.tif`

### K Layer
1. **Download SoilGrids 2.0** mean layers for **0–5, 5–15 and 15–30 cm**: `sand`, `silt`, `clay` (g/kg), `soc` (dg/kg), `cfvo` (cm³/dm³). Use the 1 km EPSG:4326 aggregated release if available, otherwise the 250 m VRTs. Depth-weight them to 0–30 cm, consistent with SOC.
2. Convert units:
   - OM % = 1.724 × SOC %, with SOC % = soc / 100.
   - Coarse fragments $R_c$ % = cfvo / 10.
3. Classify the USDA texture class per pixel, then:
   - **permeability $p$**: sand = 1; loamy sand, sandy loam = 2; loam, silt loam, silt = 3; sandy clay loam, clay loam = 4; silty clay loam, sandy clay = 5; silty clay, clay = 6 (Panagos et al. 2014 / USDA).
   - **structure $s$**: lookup by texture and OM class, following Borrelli et al. 2017 SI / Panagos et al. 2014.
   - **very fine sand**: estimated as a fixed fraction of sand. This assumption must be documented and tested for sensitivity. Note that the nomograph is valid for silt + vfs ≤ 70%.
4. Compute $K$ (nomograph) and $K_{st}=K\cdot e^{-0.04(R_c-10)}$ (for $R_c>10\%$) at native resolution. Compute $K_{EPIC}$ in parallel.
5. Aggregate both to 5′ by mean.

Output files:
- `rusle_Kst_5m.tif`
- `rusle_K_epic_5m.tif` (QA)

### LS Layer
1. **Inputs:** MERIT DEM (slope θ) and MERIT Hydro `upa` (upstream drainage area), both at 3″. Coverage runs from 60°S to 90°N, which spans the UHTH crop extent.
2. **Contributing area per cell:** $A = upa\cdot10^6 - D^2$ (m²).
   - **Cap A** to a maximum hillslope length (default $\lambda_{max}$ = 300 m, i.e. $A\le\lambda_{max}\cdot D$). Run a sensitivity test at 100–500 m.
   - Mask channel cells (upa above a threshold). This avoids the unrealistic LS seen in Fadl et al. (2025), and represents field-scale slope lengths.
3. **L:** Desmet & Govers with McCool β and $x = |\sin\alpha|+|\cos\alpha|$ (aspect α). **S:** McCool (1987), $S=10.8\sin\theta+0.03$ for tan θ < 0.09 and $S=16.8\sin\theta-0.50$ otherwise (Nearing 1997 as an option for very steep slopes).
4. **Aggregate to 5′ twice:**
   - **`LS_all`**: mean over all land. Used for forest, grassland and urban LEAFs.
   - **`LS_crop`**: cropland-weighted mean. Weights come from ESA WorldCover cropland fraction, or `ESA100_Cropland_fraction.tif` from GloSEM 1.3. Crops sit on flatter ground than the cell average, so crop LEAFs should use `LS_crop`.
5. **Process 5° tiles** (or run in Earth Engine) and write only the 5′ aggregates. Never hold a global 90 m LS raster in memory.

Output files:
- `rusle_LS_all_5m.tif`
- `rusle_LS_crop_5m.tif`

### Validation of the new layers
- Aggregate the new R, $K_{st}$ and LS to the GloSEM v1.1 0.2083° grid. Compare with `RUSLE_{R,K,LS}Factor_v1.1_25km.tif`: correlation, bias map, and ratio by continent and biome.
- Compare $K$ (nomograph) with $K_{EPIC}$.
- Compare the MFI-ratio scenarios with the published CMIP5 future-R ratios (Panagos et al. 2022). Their global average change is +26–29% by 2050.
- Re-run a handful of LEAF flows (wheat, maize, soy, coffee, grassland) with the new RLSK. Compare country, subcountry and ecoregion means with the current `se_*_clipped_2` outputs, and with GloSEM 1.3 cropland totals (12.3 Mg ha⁻¹ yr⁻¹ global mean in 2019).
- Re-check whether the current outlier clipping (log1p winsor 0.5–99.5) is still needed at 5′.

# Incorporating precipitation changes into R layer
Any precipitation scenario enters R through a **change factor** applied to the GloREDa baseline. The factor uses the Modified Fournier Index form of Arnoldus (1980):

$$\Delta_s=\frac{\sum_{i=1}^{12}\left(P_{i,s}^2/P_s\right)^{1.5}}{\sum_{i=1}^{12}\left(P_{i,h}^2/P_h\right)^{1.5}},\qquad R_s=R_{base}\cdot\Delta_s$$

$P_{i}$ is monthly precipitation and $P$ is annual precipitation, for scenario $s$ and the historical reference $h$ *of the same climate product*. The Arnoldus constants (1.735, 10^−0.08188) and the US→SI factor (17.02) cancel, so no regional calibration is needed. Properties:
- **Uniform change** of all months by a factor $k$ gives $\Delta=k^{1.5}$. For example, +10% precipitation gives +15% R, close to the ~1.7% R per 1% P sensitivity reported in the literature.
- **Seasonality changes** are captured: wetter wet seasons raise Δ more than the same total spread evenly.
- **Three ways to specify a scenario**, all through one function:
  1. a uniform percentage change;
  2. a 12-month change vector (user-defined or regional IPCC AR6 projections);
  3. full CMIP6 monthly fields.

**CMIP6 source (recommended): WorldClim 2.1 CMIP6 downscaled monthly precipitation at 5′.**
- About 23 GCMs; SSP1-2.6, 2-4.5, 3-7.0 and 5-8.5; periods 2021–2040, 2041–2060, 2061–2080 and 2081–2100.
- The historical reference is WorldClim 2.1 1970–2000 at 5′, already bias-corrected to it, so $P_h$ and $P_s$ are consistent.
- It is on the same 5′ lattice as UHTH. WorldClim covers 90°S–90°N (4320 × 2160 cells); UHTH is the subset from 90°S to 83.58°N (2083 rows). A window crop is enough; no resampling is needed.
- It is the same product family GloSEM used for future R (CMIP5 WorldClim), which keeps continuity with Borrelli et al.
- Alternative: CHELSA v2.1 CMIP6 (1 km), as used by Pyne et al.

**Outputs per SSP and period:**
- the GCM ensemble median of $R_s$;
- the 10th and 90th percentiles across GCMs (inter-model spread is about ±13% on average and −22% to +27% at the extremes, per fi-compass);
- optionally, each GCM separately, for use in the Monte Carlo below.

**Cross-check:** for RCP2.6/4.5/8.5 in 2050/2070, compare Δ with $R_{2050}/R_{base}$ from the published ESDAC future-erosivity maps (CMIP5, 19 GCMs).

# Incorporating Risk Hazards
The RUSLE A is a *long-term mean*. Real erosion is concentrated in a few years with intense rain, and drought years strip vegetation cover, raising C, before the rains return. A **stochastic (Monte Carlo) layer** on top of the deterministic R gives probabilities of high-erosion years under each climate scenario. It follows the catastrophe-modelling recipe of the fi-compass EU study and FOLU Re-WIRE (see [papers_risk_summary.md](papers_risk/papers_risk_summary.md)).

### Model
For pixel $p$, simulated year $y$, climate scenario $s$:

$$A_{p,y,s} = \underbrace{R_{base,p}\,\Delta_{p,s}}_{\text{scenario mean}}\cdot\underbrace{\varepsilon_{p,y}}_{\text{interannual}}\cdot\underbrace{h^{R}_{p,y}}_{\text{extreme rain}}\cdot K_{st,p}\cdot LS_p\cdot C_p\cdot\underbrace{h^{C}_{p,y}}_{\text{drought cover loss}}$$

| Term | Meaning | How to estimate |
|---|---|---|
| $\varepsilon_{p,y}$ | Interannual erosivity anomaly (mean 1) | Annual MFI time series from **ERA5-Land monthly precipitation (1950–present, 0.1°)**, as $\varepsilon_y = MFI_y/\overline{MFI}$, aggregated to 5′. Alternatives: CHIRPS (1981–present, 50°S–50°N only) or IMERG monthly (2000–present; the source already used for SOC). |
| $h^{R}_{p,y}$ | **Extreme-rain / flood** amplification. Monthly totals under-represent event intensity. | $h^R_y=(Rx5day_y/\overline{Rx5day})^{\gamma}$ from ERA5-Land daily data, or from the CMIP6 ETCCDI indices for the future. Default γ = 1, with sensitivity tests at 0.5–1.5. Where Rx5day exceeds the local 1-in-20-year level, this is the "flood year" state. |
| $h^{C}_{p,y}$ | **Drought** effect on cover (C rises, mainly on grassland and rainfed crops). The R reduction in dry years is already in $\varepsilon_y$. | Drought state: SPEI-12 ≤ −1.5 (global SPEIbase, or computed from precipitation − PET). Multiplier $h^C = C(NDVI_{drought})/C(NDVI_{normal})$ via van der Knijff $C=e^{-2\,NDVI/(1-NDVI)}$ (Pyne/Fadl), calibrated from MODIS NDVI anomalies in historical drought years. It carries over one year (drought followed by intense rain = compound case). Irrigated land uses get $h^C=1$. |

### Sampling scheme
1. **Historical year library** (≈1981–2023). For each year, store whole-field rasters of $\varepsilon_y$, $h^R_y$ and drought state. Resampling **entire years** keeps the spatial correlation between pixels, so that regional (country or ecoregion) aggregates of simulated years are realistic.
2. **Scenario conditioning by re-weighting.**
   - For each scenario and period, CMIP6 gives the projected frequency of drought years, $p_s(\text{drought})$ (SPEI), and of flood years, $p_s(\text{flood})$ (Rx5day above the baseline 1-in-20 level). Future windows are 20–21 years, centred on the target year as in the CIAT method.
   - Historical years get weights $w_y = p_s(\text{state}_y)/p_h(\text{state}_y)$. The bootstrap then reproduces future hazard frequencies while keeping observed spatial patterns.
   - Intensity changes enter through $\Delta_s$ (mean) and an optional Rx5day intensity scaling.
3. **Monte Carlo** of **N = 10,000 simulated years** per scenario, following fi-compass. The draws are years from the weighted library.
   - An outer loop over parameter uncertainty: GCM choice, γ, the $h^C$ calibration, and optionally K/LS uncertainty.
   - Because the model is multiplicative and K, LS and C are static, the per-pixel distribution follows directly from the weighted year library. Tail statistics can be computed as weighted quantiles without storing 10,000 global rasters; only the regional aggregates need explicit simulation.
4. **Tails beyond the record** (> 1-in-40 years): fit a Generalised Pareto Distribution to the peaks-over-threshold of simulated $A$ (as in fi-compass), or a lognormal to annual $\varepsilon\cdot h^R$.

### Outputs (per land-use flow, scenario and period; rasters at 5′ and regional LEAF tables)
- **Mean annual soil loss** (AAL analogue). It should match the deterministic LEAF ($R_{base}\Delta_s\cdot K_{st} LS\, C$) when $E[\varepsilon h^R h^C]\approx1$, which is a useful check.
- **1-in-10 and 1-in-50-year soil loss** (PML analogues; 10% and 2% annual probability).
- **Exceedance probability** of a tolerance, e.g. $P(A>10\ \text{t ha}^{-1}\text{yr}^{-1})$, and the exceedance curve.
- **Management as "Adapt" scenarios:** the same simulation with reduced tillage, residues and cover crops (via $C_{mgmt}$) gives the *avoided* mean and extreme erosion per practice.
- **Optional impact link:** % change in erosion → % change in crop yield gap via the Hadi et al. (2026) elasticity (0.044 globally; local causal-forest values in hotspots), then to production or value loss, in the FOLU "risk-to-resilience" framing.

# Data to download
| Layer | Dataset | Resolution | Source | Status |
|---|---|---|---|---|
| R baseline | GloREDa global rainfall erosivity (annual) and monthly R-factor (12 rasters) | 30″ (~1 km) | ESDAC (request form; Panagos et al. 2017, 2023). The ESDAC GloREDa page lists an access expiry of 14 Aug 2026, so access may need re-requesting. Data may not be passed to third parties. Monthly maps also on Zenodo (record 8036998). | To download |
| R scenario cross-check | Global rainfall erosivity projections 2050/2070, RCP2.6/4.5/8.5 (19 GCMs) | 30″ | ESDAC (Panagos et al. 2022) | To download |
| R scenarios | WorldClim 2.1 historical (1970–2000) and CMIP6 future monthly `prec` | 5′ | worldclim.org | To download |
| Stochastic R | ERA5-Land monthly and daily total precipitation (1950–present) | 0.1° | Copernicus CDS | To download |
| Future hazards | CMIP6 ETCCDI indices (Rx5day, R95p) and/or daily precipitation; SPEI | 0.25°–1° | Copernicus CDS (CMIP6 climate extreme indices), CIL GDPCIR, SPEIbase | To download |
| K | SoilGrids 2.0 `sand`, `silt`, `clay`, `soc`, `cfvo` at 0–5 / 5–15 / 15–30 cm | 250 m (or 1 km aggregated) | files.isric.org | To download (sand and clay 15–30 cm already on disk at 5′) |
| LS | MERIT DEM and MERIT Hydro (`upa`) | 3″ (~90 m) | U. Tokyo (free registration), or Google Earth Engine catalog | To download |
| LS crop weights | ESA WorldCover 2021, or GloSEM 1.3 `ESA100_Cropland_fraction.tif` | 10–100 m | ESA / ESDAC | To download |
| Drought C calibration | MODIS NDVI (MOD13A3 monthly) | 1 km | NASA LP DAAC / GEE | Optional |

# Implementation plan
1. **New module `src/sbtn_leaf/rusle.py`** (keeping the style of `map_calculations.py`) with:
   - `r_factor_change_mfi(prec_hist, prec_scen)`
   - `k_factor_nomograph(...)`, `k_factor_epic(...)`, `stone_correction(...)`
   - `ls_factor_desmet_govers(slope, upa, aspect, lambda_max)`
   - `aggregate_to_grid(src, template, weights=None)`: windowed mean or weighted mean onto the UHTH grid
   - `simulate_erosion(...)`: weighted year library, Monte Carlo, exceedance statistics
2. **Notebook `RUSLE/RUSLE_layers_5arcmin.ipynb`:** download and process R, K and LS; validate against GloSEM v1.1; write `rusle_*_5m.tif` and the new `rlsk_5m_<scenario>.tif`.
3. **Re-run LEAFs** with the existing pipeline (`build_cfs_gpkg_from_rasters`) using the 5′ C rasters directly, with no 25 km resampling. Compare with the current outputs.
4. **Notebook `RUSLE/RUSLE_scenarios_hazards.ipynb`:** CMIP6 Δ maps, the year library, Monte Carlo, and AAL/PML-type outputs for a few focal flows first (wheat, maize, soy, coffee, grassland), then all flows.
5. **Update [documentation/Soil_Erosion_Documentation.md](../documentation/Soil_Erosion_Documentation.md)** with the new sources, citations and resolution (and fix the TODOs listed in [RUSLE_so_far.md](RUSLE_so_far.md)).

# Open questions / risks
- **GloREDa licence:** "cannot be passed to third parties". Check whether derived 5′ R layers and LEAFs may be published.
- **Very fine sand and structure lookups for K1:** if the Borrelli et al. (2017) SI approach cannot be reproduced, use K2 (EPIC) as primary.
- **LS slope-length cap:** LS is sensitive to $\lambda_{max}$. Choose the value by comparison with GloSEM v1.1 LS (and EU RUSLE2015 LS for Europe).
- **The MFI ratio** captures monthly totals, not event intensity. The $h^R$ (Rx5day) term and its γ are the main stochastic assumptions and need sensitivity tests.
- **Drought–cover coupling** ($h^C$) is a new extension of RUSLE practice. Report results with and without it.

# References
- Arnoldus, H.M.J. (1980). An approximation of the rainfall factor in the USLE. In *Assessment of Erosion*, Wiley.
- Borrelli, P. et al. (2017). An assessment of the global impact of 21st century land use change on soil erosion. *Nat. Commun.* 8, 2013.
- Borrelli, P. et al. (2022). GloSEM: High-resolution global estimates of present and future soil displacement in croplands by water erosion. *Sci. Data* 9, 406.
- Desmet, P. & Govers, G. (1996). A GIS procedure for automatically calculating the USLE LS factor on topographically complex landscape units. *J. Soil Water Conserv.* 51, 427–433.
- Fadl, M.E. et al. (2025). Integrating RUSLE, AHP, GIS, and cloud-based geospatial analysis for soil erosion assessment under Mediterranean conditions. *Sci. Rep.* 15, 38494.
- fi-compass (2025). *Insurance and Risk Management Tools for Agriculture in the EU — Market analysis.* EIB/EC.
- FOLU (2026). *Re-WIRE — Agricultural Resilience: From cost to investment opportunity.*
- Hadi, H. et al. (2026). Land degradation is associated with larger crop yield gaps across global croplands. *Nat. Food* 7, 678–687.
- McCool, D.K. et al. (1987, 1989). Revised slope steepness / slope length factors for the USLE. *Trans. ASAE*.
- Panagos, P. et al. (2014). Soil erodibility in Europe: a high-resolution dataset based on LUCAS. *Sci. Total Environ.* 479–480, 189–200.
- Panagos, P. et al. (2015). Estimating the soil erosion cover-management factor at the European scale. *Land Use Policy* 48, 38–50.
- Panagos, P. et al. (2017). Global rainfall erosivity assessment based on high-temporal resolution rainfall records. *Sci. Rep.* 7, 4175.
- Panagos, P. et al. (2022). Global rainfall erosivity projections for 2050 and 2070. *J. Hydrol.* 610, 127865.
- Panagos, P. et al. (2023). Global rainfall erosivity database (GloREDa) and monthly R-factor data at 1 km spatial resolution. *Data in Brief* 50, 109482.
- Pyne, B. et al. (2026). Analyzing soil erosion trends and future predictions using the RUSLE model: Dwarakeswar River Basin, West Bengal. *ISWCR* 14, 100568.
- Sharpley, A.N. & Williams, J.R. (1990). EPIC — Erosion/Productivity Impact Calculator. USDA Tech. Bull. 1768.
- Van der Knijff, J.M., Jones, R.J.A. & Montanarella, L. (2000). *Soil erosion risk assessment in Europe.* EUR 19044 EN.
- Wischmeier, W.H. & Smith, D.D. (1978). *Predicting rainfall erosion losses.* USDA Agriculture Handbook 537.
- Zhang, H. et al. (2025). Assessing the impact of extreme climate events on European gross primary production. *Agric. For. Meteorol.* 362, 110374.
