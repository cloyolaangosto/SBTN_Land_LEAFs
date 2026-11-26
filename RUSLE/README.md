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
| K1 | **Wischmeier & Smith (1978) nomograph** + stone correction | Borrelli et al. 2017, 2022 (GloSEM 1.1 and 1.3: nomograph only, printed with 0.137, no stone correction); stone correction from Panagos et al. 2014 | $K=\frac{2.1\cdot10^{-4}M^{1.14}(12-OM)+3.25(s-2)+2.5(p-3)}{100}\cdot0.1317$, with $M=(\%silt+\%vfs)(100-\%clay)$; $K_{st}=K\cdot e^{-0.04(R_c-10)}$ for $R_c>10\%$ | Sand, silt, clay, OM, coarse fragments (SoilGrids); structure class $s$ and permeability class $p$ from texture-class lookups; very fine sand ($vfs$) estimated (not in SoilGrids) | Medium (lookups and a vfs assumption) |
| K2 | **Williams / EPIC** (Sharpley & Williams 1990) | Pyne et al. 2026 | $K=[0.2+0.3e^{-0.0256Sd(1-St/100)}]\,[\frac{St}{Cy+St}]^{0.3}\,[1-\frac{0.25OC}{OC+e^{3.72-2.95OC}}]\,[1-\frac{0.7SN}{SN+e^{-5.51+22.9SN}}]\times0.1317$ | Sand, silt, clay, OC (SoilGrids) | **Low**, pure raster algebra |
| K1b | **RUSLE2-K** (Renard & Dabney 2013) + optional global correction | Yang et al. 2024 | Nomograph with $M_{vfs}=(0.74-0.62\,sand/100)\,sand$ and a flattened texture term above 68% silt + vfs; corrected $K=2.1311K_{RUSLE2}-0.0359$ (144 measured sites, R² 0.64) | Same as K1; Yang take $s$ from a WRB soil-type map (lookup not published) | Low once K1 inputs exist (implemented: `rusle.k_factor_rusle2`, `rusle.build_k_rusle2`) |
| K3 | **Renard et al. (1997) geometric mean diameter** | Fadl et al. 2025 | $K=0.0034+0.0405\exp[-0.5(\frac{\log D_g+1.659}{0.7101})^2]$ | Sand, silt, clay | Very low, but ignores OM and stones |

### Methods — LS (slope length and steepness)
| # | Method | Paper | Formula / approach | Inputs | Effort |
|---|---|---|---|---|---|
| LS1 | **Desmet & Govers (1996)** 2-D unit contributing area | Borrelli et al. 2022 | $L_{i,j}=\frac{(A_{i,j}+D^2)^{m+1}-A_{i,j}^{m+1}}{D^{m+2}\,x_{i,j}^m\,22.13^m}$ with $m=\beta/(1+\beta)$, $\beta=\frac{\sin\theta/0.0896}{3\sin^{0.8}\theta+0.56}$; S from McCool (1987) | 90 m DEM and upslope contributing area (SRTM/ASTER in GloSEM; SAGA, 870 tiles; slope length ≤ 122 m, slope ≤ 26.6°) | High if flow routing is done in-house. **Medium if precomputed flow accumulation is used** (MERIT Hydro) |
| LS2 | **Slope-class exponent + McCool S** (Liu steep-slope variant) | Pyne et al. 2026 | $L=(\lambda/22.13)^m$, $m$ = 0.2/0.3/0.4/0.5 by slope class; S = 10.8 sinθ + 0.03 (< 9%), 16.8 sinθ − 0.5 (9–18%), 21.9 sinθ − 0.96 (≥ 18%) | DEM, flow length | Medium. **Implemented** from the same MERIT inputs (λ = `upa`/D ≤ 122 m): `area_term="ls2"`, see [LS Layer](#ls-layer) item 9 |
| LS3 | **Cumulative slope length + CSLE** (LS-WPC; the method of the DS-LS-GS1 global 30 m LS) | Sun et al. 2024 ([LS_2026.pdf](papers/LS_2026.pdf)) | D8 on a sink-filled DEM, cell sizes from latitude (no projection); slope length accumulated along the flow path until a slope cut-off (slope drop > 70% below 5% slope, > 50% above) or a channel (area > 0.8 km²); $L=\frac{\lambda_{out}^{m+1}-\lambda_{in}^{m+1}}{(\lambda_{out}-\lambda_{in})22.13^m}$, $m$ = 0.2/0.3/0.4/0.5 by slope; S = 10.8 sinθ + 0.03 (< 5°), 16.8 sinθ − 0.50 (5–10°), 21.91 sinθ − 0.96 (≥ 10°). No length cap, no slope clamp | DEM only (flow routed in-house) | Medium. **Implemented**: `sbtn_leaf.rusle_ls3` (numba), see [LS Layer](#ls-layer) item 10 |
| LS4 | **Shinde et al. (2011)** (listed as LS3 before 2026-11-25) | Fadl et al. 2025 | $LS=(L/22.1)^m(65.4\sin^2\theta+4.56\sin\theta+0.0065)$ | DEM, slope length | Low, but uncapped lengths give unrealistic LS (up to 4,110 in the paper) |

### Selected K and LS methods
- **K → K1 (GloSEM nomograph + stone correction), with K2 (EPIC) computed alongside as a check.**
  - K1 is the nomograph GloSEM 1.1 and 1.3 use, so the new layer stays comparable with the existing LEAFs.
  - Both GloSEM papers (Borrelli et al. 2017 SM Eq. 11; 2022 Eq. 4) print the conversion factor as 0.137. The standard US→SI factor, used by Panagos et al. (2014) and here, is 0.1317, about 4% lower.
  - Neither GloSEM paper gives a stone correction. $K_{st}$ comes from Panagos et al. (2014, Europe, surface stone cover) and is our addition. GloSEM 1.3 lists coarse fragments among its SoilGrids inputs without saying how they enter.
  - GloSEM v1.1 took its soil properties from SoilGrids1km (Hengl et al. 2014), not SoilGrids 2.0. Part of the low agreement with the v1.1 K layer (r ≈ 0.4) may come from that difference in data version.
  - K2 is almost free once the SoilGrids layers are downloaded. Where the two disagree strongly, the nomograph assumptions (vfs, $s$, $p$) can be flagged.
  - If the vfs and structure lookups cannot be reproduced defensibly, **fall back to K2**. It is the easiest method fully supported by open data.
  - **K is floored at `K_MIN` = 0.004** t ha h ha⁻¹ MJ⁻¹ mm⁻¹ in `calculate_K_RUSLE_2` and `calculate_K_Epic`, because K = 0 would mean no erosion at all. In RUSLE2, negative structure and permeability terms gave 0.007% negative pixels. 0.004 is the lowest value of the EU K map (Panagos et al. 2014, range 0.004–0.076) and the lowest measured benchmark soil (Wischmeier & Smith 1978: 0.03 US units × 0.1317). It also matches the floor of the Renard $D_g$ equation (0.0034; Yang et al. 2024 Eq. 13) and the GloSEM v1.1 25 km minimum (0.0047). K EPIC is kept **uncorrected**: Yang et al.'s Eq. 27 correction was negative in 26% of pixels and has been dropped (see [papers_summary.md](papers/papers_summary.md), Yang caveats).
- **LS → LS1 (Desmet & Govers, as GloSEM), using precomputed MERIT Hydro flow accumulation and MERIT DEM slope at 3″ (~90 m), aggregated to 5′.**
  - This reproduces the GloSEM method without running global flow routing.
  - It runs locally, one 30° MERIT archive at a time (see [LS Layer](#ls-layer)). Google Earth Engine also hosts both datasets, but it was dropped: it needs a registered Cloud project, and free use is limited by compute quotas.

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
3. Classify the USDA texture class per pixel, then (see [papers_summary.md §4–5](papers/papers_summary.md); implemented in `rusle.build_structure_permeability_vfs`):
   - **permeability $p$**: sand = 1; loamy sand, sandy loam = 2; loam, silt loam, silt = 3; sandy clay loam, clay loam = 4; silty clay loam, sandy clay = 5; silty clay, clay = 6 (Panagos et al. 2014, Table 2). Then **calibrate with coarse fragments** as Panagos did: $K_b = K_f(1-R_w)$ (Brakensiek et al. 1986), with $K_f$ the Rawls et al. (1982) conductivity of the texture class, and reclassify $p$ from $K_b$. This only makes $p$ slower. In the EU, it raised the dominant $p$ from 3 to 4 and K by ~15%.
   - **structure $s$**: Panagos et al. took it from the ESDB, which is EU-only. GloSEM derives it from texture (Borrelli et al. 2017), but the table is not published. Default: a texture lookup (our assumption: sand, loamy sand 1; sandy loam, loam, silt loam, silt 2; sandy clay loam, clay loam, silty clay loam 3; sandy clay, silty clay, clay 4). Alternative: constant $s$ = 2 (AH 537 "fine granular topsoil"; the Panagos dominant value). Sensitivity is ~2.5% of K.
   - **very fine sand**: 20% of total sand (Panagos et al. 2014; 33.3% gives +11% K).
4. Compute $K$ (nomograph) with the Panagos caps: **silt ≤ 70%** (the nomograph is valid for silt + vfs ≤ 70%; Panagos capped silt at 70% and dropped > 80%), and **OM ≤ 4%**. Then compute $K_{st}=K\cdot e^{-0.04(R_c-10)}$ (for $R_c>10\%$) at native resolution. Panagos's $R_c$ is *surface* stone cover; SoilGrids `cfvo` is *subsurface* volume, so using it for $St$ is a proxy. Compute $K_{EPIC}$ in parallel.
5. Aggregate both to 5′ by mean.

Output files:
- `rusle_Kst_5m.tif`
- `rusle_K_epic_5m.tif` (QA)

### LS Layer
1. **Inputs:** MERIT DEM (slope θ) and MERIT Hydro `upa` (upstream drainage area), both at 3″. Coverage runs from 60°S to 90°N, which spans the UHTH crop extent.
2. **Contributing area per cell:** $A = upa\cdot10^6 - D^2$ (m²).
   - **Cap A** to a maximum hillslope length (default $\lambda_{max}$ = 122 m, i.e. $A\le\lambda_{max}\cdot D$). 122 m is the longest USLE plot length and the GloSEM v1.1 setting (Borrelli et al. 2017 SM). Run a sensitivity test at 100–500 m.
   - Mask channel cells (upa above a threshold). This avoids the unrealistic LS seen in Fadl et al. (2025), and represents field-scale slope lengths.
3. **L:** Desmet & Govers with McCool β and $x = |\sin\alpha|+|\cos\alpha|$ (aspect α). **S:** McCool (1987), $S=10.8\sin\theta+0.03$ for tan θ < 0.09 and $S=16.8\sin\theta-0.50$ otherwise (Nearing 1997 as an option for very steep slopes).
   - θ is **clamped at 26.6° (50%)** before m and S, as GloSEM v1.1 (after Panagos et al. 2015). Steeper cells take the clamp value; they are not masked. The SM wording ("estimations of LS were to a maximum slope angle of 26.6°") would also allow masking. The clamp is the usual reading.
   - Aspect term: the GloSEM SM prints $x=\sin\alpha+\cos\alpha$. The absolute values are Desmet & Govers's original form.
4. **Aggregate twice** (both layers are kept, to be tested against each other). LS per pixel is terrain only; the weighting acts only in the aggregation.
   - **`LS_all`**: mean over all valid land pixels.
   - **`LS_crop`**: cropland-weighted mean. Weights come from `ESA100_Cropland_fraction.tif` (GloSEM 1.3, one global 100 m file). ESA WorldCover at 10 m was rejected: it is about 1 TB on 3° tiles that do not line up with MERIT's 5° tiles. Crops tend to sit on flatter ground than the cell average, and GloSEM computes erosion on cropland pixels only.
   - **Resolution:** any whole multiple of 3″ on a lat/lon grid anchored at (−180°, 90°): 5′ by default, 30″ (≈ 1 km) or 3″ (≈ 90 m, no aggregation) for regional runs. A bounding box limits processing and output to a region; fine resolutions need one. Metric grids (true 1,000 m or 100 m) are a later step, reprojecting the finished rasters.
5. **Process one 30°×30° archive pair at a time** (DEM tar + `upa` tar of the same box). Implemented in `src/sbtn_leaf/rusle_ls.py` (`run_local`) and run from `RUSLE/notebooks/K_LS_Creation.ipynb`.
   - **Steps per box:** download both tars by hand; `run_local` reads the 36 tiles (5° each) in place through GDAL `/vsitar/` (nothing is extracted) and writes a file of sums and counts (`LS_sum`, `n`, `LSw_sum`, `w_sum`; 360×360 at 5′); check it, then delete the tars by hand. The code never downloads or deletes files. Peak disk use is one pair, a few GB. A manifest makes the run resumable and lists the boxes still to download; one parts folder holds one set of run settings (resolution, bbox, λ_max, crop or not).
   - **Tile edges:** `upa` is already global, so no flow routing crosses tiles. Slope uses central differences, which need only the 4 direct neighbours, so tile borders inside an archive are stitched exactly. At the outer 30° edges and at coasts the difference falls back to one-sided; the notebook's edge check measures the effect. 5° = 60 × 5′, so no 5′ cell straddles two tiles.
   - Never hold a global 90 m LS raster in memory; only the sums on the output grid are kept, and they are mosaicked exactly at the end.
   - **Unattended global run:** `python RUSLE/scripts/run_ls_global.py`.
     - It logs in to the two MERIT Dropbox links. Passwords go in a git-ignored `.env` at the repo root, as `MERIT_HYDRO_PASSWORD` and `MERIT_DEM_PASSWORD`.
     - Then, box by box, it downloads the DEM and `upa` tars, computes both LS versions in one pass (`area_term=("area", "sca")`) and deletes the tars.
     - Finally it mosaics both versions: `rusle_LS_all_5m.tif` (default) and `rusle_LS_all_5m_glosem_compat.tif`.
     - It is resumable through the manifests. `--dry-run` lists what would run, and `--boxes` limits the run to some boxes. The log is written to `RUSLE/outputs/ls_global_run.log`.
6. **Known differences from GloSEM.** The LS settings come from the v1.1 Supplementary Methods (Borrelli et al. 2017). The GloSEM 1.3 paper (Borrelli et al. 2022) describes the same run (SRTM/ASTER at 3″, SAGA, 870 tiles) and does not restate the parameters.
   - **Flow routing and DEM.** GloSEM ran SAGA with D-infinity / multiple-flow routing (the SM names both) on hole-filled SRTM, with ASTER GDEM v2 in the far north. We use MERIT Hydro `upa` (D8) and the MERIT DEM.
   - **25 m rescaling.** GloSEM rescaled its 90 m LS "to the values of a 25 m DEM through a linear relationship (R² = 0.998)". The coefficients are not published, so this step is not reproduced.
   - **Resampling.** GloSEM resampled LS to 250 m by nearest neighbour, which explains the speckle in the 25 km v1.1 file. Our 5′ value is the mean of all 3″ pixels. The speckle does not bias levels: the 25 km file reproduces GloSEM's published continental means (SM Fig. 14c) within about 10% (e.g. Europe 1.31 vs 1.25, Africa 0.98 vs 1.0), so it is a fair reference.
   - **Remaining offset, probably units on GloSEM's side.** With GloSEM's settings our LS is still 2.6× the v1.1 25 km layer on box `n30e000` (ratio of means; 3.4× at 300 m without the clamp). Rank correlation is 0.76, so the pattern agrees and the gap is one of level.
     - Our units follow Desmet & Govers (1996) Eq. 9: A in m², D in m, checked against the MERIT data. It is not a projection error: the ratio follows terrain, not 1/cos(latitude).
     - SAGA's LS tools, which GloSEM used, take the *specific* catchment area (m) as their area input. SAGA's Desmet & Govers code adds the cell area (m²) to it.
     - Emulating that brings the box ratio to 1.3, with the same rank correlation.
     - This is not proven: GloSEM does not state the tool or options. Details are in [papers_summary.md](papers/papers_summary.md), sections 7–8.
7. **GloSEM-compatible switch (`area_term`).** `ls_factor_desmet_govers`, `process_archive` and `run_local` take `area_term="area"` (default) or `"sca"`. The notebook sets it with `AREA_TERM`, and `"sca"` runs go to their own parts folder (`ls_parts_glosem_sca`).
   - **`"area"` (correct).** The A in Desmet & Govers Eq. 9 is the contributing area at the cell inlet, in m². With the 122 m cap, a cell on an open hillside has A = 122 m × D ≈ 9,500 m² (D ≈ 78 m at 45°N). L then reflects up to 122 m of slope draining onto the cell.
   - **`"sca"` (GloSEM/SAGA-compatible).** A is replaced by the specific catchment area A / (D·x), a length (here 122 m), and the formula still adds D² ≈ 6,100 m² to it. The number 122 is read as 122 m², i.e. about 1.6 m of slope above the cell. Every cell is then scored like a hilltop cell, and L ≈ (D / (22.13·x))^m whatever drains into it.
   - **How large the difference is.** Cells with nothing upslope get the same L in both modes. For a cell at the 122 m cap the ratio grows with slope, because the exponent m does:

     | Slope | m | L, `"area"` | L, `"sca"` | Ratio |
     |---|---|---|---|---|
     | 1% | 0.15 | 1.54 | 1.22 | 1.26 |
     | 3% | 0.31 | 2.42 | 1.51 | 1.61 |
     | 9% | 0.50 | 4.05 | 1.93 | 2.10 |
     | 20% | 0.61 | 5.45 | 2.23 | 2.44 |
     | ≥ 50% (clamp) | 0.70 | 6.81 | 2.49 | 2.73 |

     At 9%, a cell with one cell (78 m) upslope gets a ratio of 1.8. Averaged over real terrain, `"sca"` is about half of `"area"`, the size of the gap to GloSEM v1.1.
   - **What `"sca"` loses.** It ignores flow convergence, the very effect Desmet & Govers built the method for: hollows and long slopes get the same L as ridges. It also makes the slope-length cap almost irrelevant, and L depends on the grid size (∝ D^m). Use it only for continuity with GloSEM v1.1 or as a sensitivity layer, never as the main LS.
   - **Reference means.** GloSEM v1.1 mean LS by continent (250 m, SM Fig. 14c, read off the chart): Africa ≈ 1.0, Asia ≈ 2.5, Europe ≈ 1.25, North America ≈ 1.75, Oceania ≈ 0.8, South America ≈ 1.65.
8. **Global comparison with GloSEM v1.1 (full run, 2026-11-24).** All 57 MERIT boxes were processed, with no failures. Both global layers were averaged onto the GloSEM v1.1 25 km grid, and every value below is an area-weighted mean. Continents come from Natural Earth, with Russia east of 60°E counted as Asia. Coverage is 100% of GloSEM's land cells. Table: `RUSLE/outputs/ls_global_vs_glosem_by_continent.csv`.

   | Continent | GloSEM, SM Fig. 14c (250 m) | GloSEM 25 km | Default (`"area"`) | Default / GloSEM | GloSEM-compatible (`"sca"`) | GloSEM-compatible / GloSEM | Spearman |
   |---|---|---|---|---|---|---|---|
   | Africa | 1.00 | 0.98 | 2.24 | 2.29 | 1.32 | 1.35 | 0.71 |
   | Asia | 2.50 | 2.72 | 6.53 | 2.40 | 3.33 | 1.22 | 0.81 |
   | Europe | 1.25 | 1.31 | 3.28 | 2.50 | 1.70 | 1.29 | 0.68 |
   | North America | 1.75 | 1.72 | 4.27 | 2.48 | 2.19 | 1.27 | 0.74 |
   | Oceania | 0.80 | 0.90 | 1.95 | 2.17 | 1.11 | 1.24 | 0.62 |
   | South America | 1.65 | 1.62 | 3.89 | 2.40 | 2.17 | 1.33 | 0.75 |
   | **World** | – | 1.79 | **4.30** | **2.40** | **2.27** | **1.27** | 0.76 |

   - **The gap is nearly constant across continents:** 2.2–2.5× for the default and 1.2–1.35× for the GloSEM-compatible layer. A gap that steady points to how the method was applied (item 6), not to terrain or latitude.
   - **About 27% remains with `"sca"`.** It plausibly comes from the unpublished 25 m rescale, SRTM versus MERIT, and D-infinity versus D8 routing. This is not tested.
   - **The spatial pattern agrees:** Spearman correlation is 0.62–0.81, and Pearson r is about 0.75 for both layers. The continents rank in the same order in all layers.
   - The notebook's section "Global comparison of all LS layers with GloSEM v1.1" reproduces this table, with maps and scatter plots, for every mosaicked layer.
9. **LS2 (Pyne et al. 2026), alternative method, implemented with no extra data.** `area_term="ls2"` in `ls_factor_desmet_govers`, `process_archive` and `run_local`. The global script computes it in the same pass as the other two layers: parts go to `ls_parts_ls2`, and the mosaic is `rusle_LS_all_5m_ls2.tif`.
   - $L=(\lambda/22.13)^m$, with $m$ = 0.2, 0.3, 0.4, 0.5 for slopes < 1%, 1–3%, 3–9%, ≥ 9%.
   - $S$ is McCool (1987) with the Liu steep-slope branch: 10.8 sin θ + 0.03 (< 9%), 16.8 sin θ − 0.50 (9–18%), 21.9 sin θ − 0.96 (≥ 18%).
   - **Slope length.** λ = `upa` / D, the upslope area including the cell divided by the cell width: the slope length to the cell outlet. It is capped at the same 122 m, and the slope clamp at 26.6° also applies. There is no aspect term.
   - **What the cap does.** On a 90 m grid with a 122 m cap, λ only ranges from D (≈ 80 m, a hilltop cell) to 122 m. L is then close to a function of the slope class, and most of LS2's level comes from S.
   - **Test box `n30e000`:** 1.75× GloSEM v1.1, between LS1 (2.58×) and the GloSEM-compatible layer (1.33×), with the same Spearman (0.76). Its global numbers need the global run: `python RUSLE/scripts/run_ls_global.py --variants ls2` (or the notebook's run cell).
   - **Rerunning the global layers.** `--variants` picks the layers, and only boxes not yet done for them are downloaded. `--fresh` redoes everything, after renaming the old parts folders to `*_old_<time>`.
10. **LS3 (Sun et al. 2024), the DS-LS-GS1 method, implemented on MERIT.** Module `src/sbtn_leaf/rusle_ls3.py`, using numba (`pip install .[rusle]`); notebook section "LS3" in `K_LS_Creation.ipynb`.
    - **Method:** a sink-filled MERIT DEM (priority-flood) and D8 steepest descent, with cell sizes from latitude: no projection, the "LS-WPC" idea.
      - Slope length accumulates along the flow path ("Max" mode). It ends at a slope cut-off (slope drops by > 70% where the slope is < 5%, by > 50% above that) or at a channel (D8 area > 0.8 km²). These are the LS-TOOL defaults.
      - L is the segment formula with m = 0.2/0.3/0.4/0.5 by slope class. S is the CSLE: 10.8 sinθ + 0.03 below 5°, 16.8 sinθ − 0.50 at 5–10°, 21.91 sinθ − 0.96 from 10°. The paper prints −0.05 in the middle branch, a typo; −0.50 keeps S continuous at 5°.
      - There is no slope-length cap and no slope clamp.
    - **Only the DEM is needed:** flow is routed in-house, so each box needs just its MERIT DEM tar. Tiles are 5°, with a 0.1° buffer from the neighbouring tiles; a 240 px buffer changes the 5′ means by < 0.01%. It takes about 30 s per tile, 10–13 min per 30° box.
    - **Synthetic check:** on the paper's Himmelblau–Orlandini surface, LS3 reproduces its Table 1. Slope mean is 51.0° vs 50.9° (max 84.97° in both), slope length mean 67.8 vs 65.0 m, LS mean 37.7 vs 36.9 (`tests/test_rusle_ls3.py`).
    - **Boxes `n30e000` + `n00e000`** (0–60°N, 0–30°E; 208 k 5′ cells; 2026-11-25):

      | Layer | LS3 ÷ layer (same cells) | Spearman with LS3 | Layer ÷ GloSEM v1.1 (25 km) | Median ratio to DS-LS-GS1 (39 countries) |
      |---|---|---|---|---|
      | **LS3** | 1 | 1 | **3.02** | **1.05** |
      | LS1 | 1.22 | 0.98 | 2.38 | 0.87 |
      | LS GloSEM-compatible | 2.18 | 0.98 | 1.30 | 0.51 |
      | LS2 (`n30e000` only) | 1.86 | 0.99 | 1.75 | 0.58 |

      Country means, DS-LS-GS1 / LS3 / LS1: Switzerland 25.1 / 26.2 / 16.7, Italy 13.5 / 13.9 / 10.1, Germany 3.0 / 3.2 / 2.6, Poland 1.12 / 1.27 / 1.10, Netherlands 0.13 / 0.19 / 0.17.
    - **What it shows:**
      - Run on our 3″ MERIT DEM, LS3 lands within 5% of the published 1″ SRTM DS-LS-GS1 country means, with Spearman 1.00 across 39 countries. The implementation reproduces that dataset, and DEM resolution matters less here than its Suppl. Table 1 suggested.
      - LS3 is 1.2× LS1 and 3× GloSEM v1.1. Two independent full-flow-path methods put LS well above GloSEM.
    - **Global run** (2026-11-25, all 57 boxes; `python RUSLE/scripts/run_ls_global.py --variants ls3`, about 6 h including the DEM downloads):

      | Layer | ÷ GloSEM v1.1 (25 km, world) | ÷ GloSEM by continent | ÷ DS-LS-GS1 (183 countries, area-weighted) | median country ratio to DS-LS-GS1 | Spearman with DS-LS-GS1 |
      |---|---|---|---|---|---|
      | **LS3** | **3.07** | 2.7–3.3 | **1.05** | **1.04** (1.03–1.05 by continent) | 0.99 |
      | LS1 | 2.40 | 2.2–2.5 | 0.82 | 0.86 | 0.99 |
      | LS GloSEM-compatible | 1.27 | 1.2–1.35 | 0.43 | 0.49 | 0.99 |
      | GloSEM v1.1 | 1 | 1 | 0.37 | 0.37 (0.31–0.41) | 0.94 |

      - The global run confirms the two test boxes: LS3 reproduces DS-LS-GS1 on every continent, and is 1.28× LS1 on the same 5′ cells (world mean 5.15 vs 4.02). GloSEM v1.1 is the low outlier.
      - Tables: `RUSLE/outputs/ls_global_vs_glosem_by_continent.csv` and `ls_global_country_means_vs_ds_ls_gs1.csv`.
    - **Pixel-level check:** the DS-LS-GS1 tiles `N44E6` and `N48E6` were not downloaded; the download was stopped on 2026-11-25 because the TPDC FTP gave about 1 MB/min. The notebook's last LS3 cell compares them cell by cell if they are ever put on disk.

Output files:
- `rusle_LS_all_5m.tif` (default, `"area"`)
- `rusle_LS_all_5m_glosem_compat.tif` (GloSEM-compatible, `"sca"`)
- `rusle_LS_all_5m_ls2.tif` (LS2, `"ls2"`; so far only box `n30e000`)
- `rusle_LS_all_5m_ls3.tif` (LS3, Sun et al. 2024; global); `rusle_LS_all_5m_ls3_partial.tif` (boxes `n30e000` and `n00e000` only)
- `rusle_LS_crop_5m.tif` (only when crop weights are available)

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
| K structure $s$ (Yang et al. 2024 route; optional) | SoilGrids 2.0 WRB most probable reference soil group (`wrb/MostProbable.vrt`), plus a WRB → $s$ lookup. Yang's lookup is not published: ask the authors, or build one from the ESDB pedotransfer rules (Van Ranst et al. 1995) via Panagos Table 1 | 250 m (aggregate by mode to the 1 km grid) | files.isric.org/soilgrids/latest/data/wrb/; ESDAC | Optional, replaces the texture lookup |
| K validation | Yang et al. (2024) Table A.1, 144 measured-K sites (in the PDF appendix) | Points | `RUSLE/papers/` | To extract |
| LS | MERIT DEM and MERIT Hydro (`upa`), 30°×30° tar archives | 3″ (~90 m) | U. Tokyo (free registration); downloaded manually box by box; processed by `rusle_ls.run_local` | To download (one box at a time, deleted after processing) |
| LS crop weights | GloSEM 1.3 `ESA100_Cropland_fraction.tif` | 100 m | ESDAC | To download |
| Drought C calibration | MODIS NDVI (MOD13A3 monthly) | 1 km | NASA LP DAAC / GEE | Optional |

# Implementation plan
1. **New module `src/sbtn_leaf/rusle.py`** (keeping the style of `map_calculations.py`) with:
   - `r_factor_change_mfi(prec_hist, prec_scen)`
   - `k_factor_nomograph(...)`, `k_factor_epic(...)`, `stone_correction(...)`
   - LS is in its own module, `src/sbtn_leaf/rusle_ls.py`: `ls_factor_desmet_govers`, `process_archive`, `run_local`, `mosaic_ls`
   - `aggregate_to_grid(src, template, weights=None)`: windowed mean or weighted mean onto the UHTH grid
   - `simulate_erosion(...)`: weighted year library, Monte Carlo, exceedance statistics
2. **Notebook `RUSLE/RUSLE_layers_5arcmin.ipynb`:** download and process R, K and LS; validate against GloSEM v1.1; write `rusle_*_5m.tif` and the new `rlsk_5m_<scenario>.tif`.
3. **Re-run LEAFs** with the existing pipeline (`build_cfs_gpkg_from_rasters`) using the 5′ C rasters directly, with no 25 km resampling. Compare with the current outputs.
4. **Notebook `RUSLE/RUSLE_scenarios_hazards.ipynb`:** CMIP6 Δ maps, the year library, Monte Carlo, and AAL/PML-type outputs for a few focal flows first (wheat, maize, soy, coffee, grassland), then all flows.
5. **Update [documentation/Soil_Erosion_Documentation.md](../documentation/Soil_Erosion_Documentation.md)** with the new sources, citations and resolution (and fix the TODOs listed in [RUSLE_so_far.md](RUSLE_so_far.md)).

# Open questions / risks
- **GloREDa licence:** "cannot be passed to third parties". Check whether derived 5′ R layers and LEAFs may be published.
- **Structure lookup for K1:** vfs (20% of sand) and $p$ are now sourced from Panagos et al. (2014). $s$ is still our own texture lookup, because the Borrelli et al. (2017) table is not available. Choose between it and the constant $s$ = 2 by comparison with GloSEM v1.1 K. If K1 still disagrees strongly with GloSEM, use K2 (EPIC) as primary.
- **LS slope-length cap:** LS is sensitive to $\lambda_{max}$. Choose the value by comparison with GloSEM v1.1 LS (and EU RUSLE2015 LS for Europe).
- **The MFI ratio** captures monthly totals, not event intensity. The $h^R$ (Rx5day) term and its γ are the main stochastic assumptions and need sensitivity tests.
- **Drought–cover coupling** ($h^C$) is a new extension of RUSLE practice. Report results with and without it.

# References
- Arnoldus, H.M.J. (1980). An approximation of the rainfall factor in the USLE. In *Assessment of Erosion*, Wiley.
- Brakensiek, D.L., Rawls, W.J. & Stephenson, G.R. (1986). Determining the saturated hydraulic conductivity of a soil containing rock fragments. *Soil Sci. Soc. Am. J.* 50, 834–835.
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
- Rawls, W.J., Brakensiek, D.L. & Saxton, K.E. (1982). Estimation of soil water properties. *Trans. ASAE* 25(5), 1316–1320.
- Sharpley, A.N. & Williams, J.R. (1990). EPIC — Erosion/Productivity Impact Calculator. USDA Tech. Bull. 1768.
- Van der Knijff, J.M., Jones, R.J.A. & Montanarella, L. (2000). *Soil erosion risk assessment in Europe.* EUR 19044 EN.
- Renard, K.G. & Dabney, S.M. (2013). RUSLE2 soil erodibility (K) computation, as cited in Yang et al. (2024).
- Yang, M., Yang, Q., Zhang, K., Pang, G. & Huang, C. (2024). Global soil erodibility factor (K) mapping and algorithm applicability analysis. *Catena* 239, 107943.
- Wischmeier, W.H. & Smith, D.D. (1978). *Predicting rainfall erosion losses.* USDA Agriculture Handbook 537.
- Zhang, H. et al. (2025). Assessing the impact of extreme climate events on European gross primary production. *Agric. For. Meteorol.* 362, 110374.
