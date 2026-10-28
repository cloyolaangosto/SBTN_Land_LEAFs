# RUSLE method papers — summary

Three papers are candidate sources of methods for the new R, K and LS layers. For each one: what it does, how it computes R, K and LS, what data it needs, and caveats for a global, 5′ application.

| File | Reference | Scale | Main use for us |
|---|---|---|---|
| [GLOSEM_1_3.pdf](GLOSEM_1_3.pdf) | Borrelli, Ballabio, Yang, Robinson & Panagos (2022). *GloSEM: High-resolution global estimates of present and future soil displacement in croplands by water erosion.* Sci. Data 9:406. [doi:10.1038/s41597-022-01489-x](https://doi.org/10.1038/s41597-022-01489-x) | Global, 100 m | Reference methods, consistent with current LEAFs (GloSEM v1.1) |
| [RUSLE_2_2025.pdf](RUSLE_2_2025.pdf) | Fadl et al. (2025). *Integrating RUSLE, AHP, GIS, and cloud-based geospatial analysis for soil erosion assessment under Mediterranean conditions.* Sci. Rep. 15:38494. [doi:10.1038/s41598-025-22503-3](https://doi.org/10.1038/s41598-025-22503-3) | Mitidja Plain (Algeria), 30 m | Simplest formulas; Google Earth Engine (GEE) workflow |
| [RUSLE_3.pdf](RUSLE_3.pdf) | Pyne et al. (2026). *Analyzing soil erosion trends and future predictions using the RUSLE model: A case study of the Dwarakeswar River Basin, West Bengal.* ISWCR 14:100568. [doi:10.1016/j.iswcr.2025.08.011](https://doi.org/10.1016/j.iswcr.2025.08.011) | River basin (4,341 km²), 30 m | Monthly-precipitation R (Modified Fournier Index) driven by CMIP6 scenarios; EPIC K from SoilGrids |

---

## 1. Borrelli et al. (2022) — GloSEM 1.3

**What it is.** A data descriptor for GloSEM 1.3. It gives RUSLE-based soil displacement for global croplands (~1.4 billion ha, 199 countries) at ~100 m (0.00099°), for 2019 and 2070 under SSP1-RCP2.6, SSP2-RCP4.5 and SSP5-RCP8.5. Its main change from v1.1/1.2 (250 m, MODIS-based) is the cropland mask: the ESA/Copernicus CGLS-LC100 fractional cropland layer (2019).

**R factor.** A Gaussian Process Regression (GPR) between station R values and climate covariates:

- Station data: GloREDa, 3,625 stations in 63 countries, sub-hourly or hourly pluviographs. Event erosivity (EI30) follows USDA Handbook 703, computed with the USDA RIST tool; $R = \frac{1}{n}\sum_j\sum_k (EI_{30})_k$.
- Covariates: WorldClim 2.1 at 30″, namely mean, minimum and maximum monthly precipitation, mean monthly temperature, precipitation of the wettest and driest month, and precipitation seasonality.
- Future R: the same GPR driven by 14 CMIP5 GCMs × 3 RCPs, giving 43 scenario combinations.
- Validation: R² = 0.85 on pre-2000 training data and 0.60 on post-2000 data. R is underestimated above ~10,000 MJ mm ha⁻¹ h⁻¹ yr⁻¹, and the coarse WorldClim covariates cause mismatch in complex terrain.

**K factor.** The Wischmeier & Smith (1978) nomograph equation:

$$K = \frac{2.1\times10^{-4} M^{1.14}(12-OM) + 3.25(s-2) + 2.5(p-3)}{100}\cdot 0.1317$$

- $M$ = (silt % + very fine sand %) × (100 − clay %)
- $OM$ = organic matter %
- $s$ = structure class; $p$ = permeability class, both derived from texture as in the GloSEM K map (Borrelli et al., 2017)
- Inputs from SoilGrids at 1 km: texture, organic matter, coarse fragments
- The PDF prints the constant as 0.137; the standard SI conversion is 0.1317.

**LS factor.** Two-dimensional unit-contributing-area method of Desmet & Govers (1996) on hole-filled SRTM + ASTER GDEM v2 at 3″ (~90 m). It was run in SAGA GIS from R over 870 tiles of about 500 × 500 km.

**C and P.**

- C: sub-national weighted crop C-factor from FAOSTAT crop shares (2001–2012), downscaled with Monfreda et al. harvested areas.
- P: conservation agriculture (CA) shares from AQUASTAT (54 countries; continental values elsewhere), assuming a 45–70% erosion reduction under CA.

**Uncertainty.** Error propagation combines MCMC uncertainty of the spatial predictions, CA area and effectiveness (Monte Carlo), rainfall intensity–kinetic energy relationships, and spread across the 14 GCMs. These are combined as the root sum of squares.

**Data released.** `SOIL_DISPLACEMENT_ESTIMATE_{2019, 2070_SSP1-RCP2.6, 2070_SSP2-RCP4.5, 2070_SSP5-RCP8.5}.tif` and `ESA100_Cropland_fraction.tif` (ESDAC and Figshare).

> ⚠️ **R, K and LS are not released separately in v1.3.** Only final soil displacement is published, and only on cropland. The authors say they *plan* to release the refined inputs. For our multiplicative LEAF scheme (separate C per crop and management), the inputs must come from elsewhere: GloREDa R maps, our own K and LS.

**Results.** Global cropland mean erosion is 12.3 Mg ha⁻¹ yr⁻¹ (17.2 Pg yr⁻¹) in 2019. By 2070 it reaches 14.6, 15.2 and 15.8 Mg ha⁻¹ yr⁻¹ under RCP2.6, 4.5 and 8.5, driven by rainfall erosivity alone. Results are within 5% of USDA NRI (US) and EC RUSLE2015 (EU) totals.

**Relevance.** This is the method family the current LEAFs already use, so building new layers with it keeps continuity with v1.1. It is the most rigorous option, but the GPR R model needs station data. The 1 km GloREDa maps produced with it are available, so we can use the outputs without re-fitting.

---

## 2. Fadl et al. (2025) — RUSLE + AHP on Google Earth Engine

**What it is.** A RUSLE assessment for the Mitidja Plain (north-central Algeria; Mediterranean climate, 600–800 mm yr⁻¹). All factors are computed in GEE and the factors are then weighted with the Analytical Hierarchy Process (AHP). Reported mean soil loss is 88.7 t ha⁻¹ yr⁻¹ (range 38–161); 41% of the area is classed as severe. Validation AUC is 0.87.

**R factor.** A linear regression on mean annual precipitation $X_a$ (mm), from CHIRPS (0.05°, 2001–2023), attributed to Choudhury & Nayak (2003):

$$RE = 79 + 0.363\,X_a \;(+\,89.7 \text{ as printed})$$

The equation is garbled in the PDF. Its reported RE range (584–1072 for 830–2250 mm) does not reproduce exactly with either reading. Coefficients of this form come from Indian regressions (e.g. Singh et al., 1981) and are regionally calibrated, so they should **not** be transferred globally.

**K factor.** The Renard et al. (1997) geometric-mean-diameter equation (texture only):

$$K = 0.0034 + 0.0405\exp\left[-0.5\left(\frac{\log D_g + 1.659}{0.7101}\right)^2\right],\quad D_g = \exp\left(\sum f_i \ln\frac{d_i+d_{i-1}}{2}\right)$$

- Inputs: USDA texture classes at 250 m (OpenLandMap).
- A lookup table by texture class and organic matter (OMC < 2% or > 2%) from McKague (2023) is also given.
- Reported K: 0–0.039, mean 0.0197 t ha h MJ⁻¹ mm⁻¹.

**LS factor.** Simple formula after Shinde et al. (2011), from SRTM 30 m:

$$LS = \left(\frac{L}{22.1}\right)^{m}\left(65.4\sin^2\theta + 4.56\sin\theta + 0.0065\right)$$

Lookup tables of LS by slope length (15–244 m) and slope (0–10%) are included. The reported LS range of 1.7–4110 suggests flow accumulation was not capped.

**C and P.**

- C: from NDVI (Sentinel-2) using van der Knijff et al. (2000), $C = \exp[-\alpha\,NDVI/(\beta-NDVI)]$, plus crop and tillage tables (e.g. no-till 0.25).
- P: from tables (contouring 0.5, strip cropping 0.25–0.37).

**AHP.** Pairwise-comparison weights (slope has the highest weight, 0.292; consistency ratio < 0.1). The weights multiply each factor: $A = \prod (F_i W_i)$.

> ⚠️ In a purely multiplicative model, multiplying every factor by a constant weight only rescales A uniformly, so AHP adds no spatial information to A itself. AHP is only meaningful for a susceptibility index. Not recommended for LEAFs.

**Relevance.** The lowest data needs (annual rainfall, texture, DEM) and an easy GEE implementation. But the R and K formulas are regional or crude: R ignores rainfall seasonality and intensity, and K ignores organic matter and stones. Use it only as a fallback or for sense-checks.

---

## 3. Pyne et al. (2026) — RUSLE with CMIP6 future scenarios

**What it is.** RUSLE for the Dwarakeswar River Basin (West Bengal; monsoon climate, ~1,357 mm yr⁻¹) for 1998, 2006, 2014 and 2022, plus projections for 2035 and 2050. Climate scenarios come from CMIP6 MIROC6 (labelled RCP 2.6, 4.5 and 8.5). Land use is projected with TerrSet Land Change Modeler (MLP neural network + Markov chain), validated with Kappa indices. Only R, C and P change in the future runs; K and LS are held constant (pedogenesis and geomorphic equilibrium arguments).

**R factor.** The Modified Fournier Index (MFI) of Arnoldus (1980), with $P_i$ = monthly and $P$ = annual rainfall (mm):

$$R = \sum_{i=1}^{12} 1.735\times10^{\left(1.5\log_{10}\frac{P_i^2}{P} - 0.08188\right)}$$

- Historical: IMD 0.25° gridded rainfall (1993–2022) at 42 grid points, interpolated by inverse distance weighting (IDW).
- Future: CHELSA v2 (~1 km) CMIP6 monthly precipitation for 2035 and 2050.

> ⚠️ The Wischmeier/Arnoldus coefficients give R in US customary units (hundreds of ft·tonf·in ac⁻¹ h⁻¹ yr⁻¹). Multiply by **17.02** for MJ mm ha⁻¹ h⁻¹ yr⁻¹. The paper's R values (194–438) suggest this conversion was not applied, so confirm before reuse. The paper also reports future mean "R" values (~23 MJ mm ha⁻¹ h⁻¹ yr⁻¹) that equal its soil-loss values, and it projects a 3–4× rise in soil erosion even though future rainfall and R decrease. Treat its future results with caution; the *method* is still useful.

**K factor.** The Williams/EPIC equation (Sharpley & Williams, 1990), using SoilGrids sand, silt, clay and organic carbon (250 m):

$$K = \left[0.2+0.3e^{-0.0256\,Sd\,(1-St/100)}\right]\left[\frac{St}{Cy+St}\right]^{0.3}\left[1-\frac{0.25\,OC}{OC+e^{3.72-2.95\,OC}}\right]\left[1-\frac{0.7\,SN}{SN+e^{-5.51+22.9\,SN}}\right]$$

$SN = 1 - Sd/100$; $Sd$, $St$, $Cy$ = sand, silt, clay %; $OC$ = organic carbon %. The output is in US units, so multiply by 0.1317 for SI.

**LS factor.** From SRTM 30 m in ArcGIS:

- L: $L = (\lambda/22.13)^m$, with $m$ = 0.2, 0.3, 0.4, 0.5 for slope < 1%, 1–3%, 3–9%, ≥ 9%.
- S: McCool et al. (1987), with the steep-slope adjustment of Liu & Zhang (2002) and Qiu et al. (2018):
  - $S = 10.8\sin\theta + 0.03$ for θ < 9%
  - $S = 16.8\sin\theta - 0.5$ for 9–18%
  - $S = 21.9\sin\theta - 0.96$ for ≥ 18%

**C and P.**

- C: NDVI from Landsat via van der Knijff (α = 2, β = 1) for past years; land-use lookup (agriculture 0.6, forest 0.5) for future years.
- P: USDA-SCS tables.

**Relevance.** The best template for **scenario-driven R**. The MFI needs only monthly precipitation, which we already have on the 5′ grid (IMERG) and which every CMIP6 product provides (WorldClim, CHELSA, NEX-GDDP). EPIC K needs only standard SoilGrids layers. Its weaknesses are a regional calibration (MFI–R) and a single GCM.

---

## Cross-paper comparison of R, K, LS methods

| Factor | Borrelli et al. 2022 (GloSEM 1.3) | Fadl et al. 2025 | Pyne et al. 2026 |
|---|---|---|---|
| **R** | GPR of GloREDa station R (3,625 stations) on WorldClim covariates; future via 14 GCMs × 3 RCPs (CMIP5) | Linear regression on annual P (CHIRPS) | Modified Fournier Index on monthly P (IMD; CHELSA CMIP6 MIROC6 for future) |
| R data needs | Pluviograph station data (not public); outputs public at 1 km | Annual P | Monthly P |
| R scenario-ready? | Yes (published maps, CMIP5) | Weakly (annual total only) | **Yes**, any monthly P scenario |
| **K** | Wischmeier nomograph (texture, very fine sand, OM, structure, permeability) + stones; SoilGrids 1 km | Renard $D_g$ (texture only) | Williams/EPIC (sand, silt, clay, OC); SoilGrids 250 m |
| **LS** | Desmet & Govers 2-D, SRTM/ASTER 90 m, SAGA | Shinde formula, SRTM 30 m | McCool S + Liu steep; $L=(\lambda/22.13)^m$; SRTM 30 m |
| Implementation effort (global) | High (GPR, global 90 m flow routing) | Low | Low–medium |
| Global defensibility | High | Low | Medium |
