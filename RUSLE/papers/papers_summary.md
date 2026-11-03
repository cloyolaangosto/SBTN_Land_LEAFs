# RUSLE method papers — summary

Three papers are candidate sources of methods for the new R, K and LS layers. For each one: what it does, how it computes R, K and LS, what data it needs, and caveats for a global, 5′ application. Two reference documents ([sections 4–5](#4-panagos-et-al-2014--soil-erodibility-in-europe-lucas)) define the K1 nomograph inputs (structure $s$, permeability $p$, very fine sand) that our K layer uses.

| File | Reference | Scale | Main use for us |
|---|---|---|---|
| [GLOSEM_1_3.pdf](GLOSEM_1_3.pdf) | Borrelli, Ballabio, Yang, Robinson & Panagos (2022). *GloSEM: High-resolution global estimates of present and future soil displacement in croplands by water erosion.* Sci. Data 9:406. [doi:10.1038/s41597-022-01489-x](https://doi.org/10.1038/s41597-022-01489-x) | Global, 100 m | Reference methods, consistent with current LEAFs (GloSEM v1.1) |
| [RUSLE_2_2025.pdf](RUSLE_2_2025.pdf) | Fadl et al. (2025). *Integrating RUSLE, AHP, GIS, and cloud-based geospatial analysis for soil erosion assessment under Mediterranean conditions.* Sci. Rep. 15:38494. [doi:10.1038/s41598-025-22503-3](https://doi.org/10.1038/s41598-025-22503-3) | Mitidja Plain (Algeria), 30 m | Simplest formulas; Google Earth Engine (GEE) workflow |
| [RUSLE_3.pdf](RUSLE_3.pdf) | Pyne et al. (2026). *Analyzing soil erosion trends and future predictions using the RUSLE model: A case study of the Dwarakeswar River Basin, West Bengal.* ISWCR 14:100568. [doi:10.1016/j.iswcr.2025.08.011](https://doi.org/10.1016/j.iswcr.2025.08.011) | River basin (4,341 km²), 30 m | Monthly-precipitation R (Modified Fournier Index) driven by CMIP6 scenarios; EPIC K from SoilGrids |
| [Panagos 2014 - soil erodability in europe.pdf](Panagos%202014%20-%20soil%20erodability%20in%20europe.pdf) | Panagos, Meusburger, Ballabio, Borrelli & Alewell (2014). *Soil erodibility in Europe: A high-resolution dataset based on LUCAS.* Sci. Total Environ. 479–480:189–200. [doi:10.1016/j.scitotenv.2014.02.010](https://doi.org/10.1016/j.scitotenv.2014.02.010) | EU-25, 500 m | How to derive $s$, $p$ and vfs for the nomograph; coarse-fragment and stone corrections; sensitivities |
| [Yang 2024 - Global soil erodability.pdf](Yang%202024%20-%20Global%20soil%20erodability.pdf) | Yang, Yang, Zhang, Pang & Huang (2024). *Global soil erodibility factor (K) mapping and algorithm applicability analysis.* Catena 239:107943. [doi:10.1016/j.catena.2024.107943](https://doi.org/10.1016/j.catena.2024.107943) | Global, SoilGrids 250 m | Compares USLE-K, RUSLE2-K, EPIC-K and Dg-K against 144 measured-K sites; RUSLE2-K fits best; gives correction equations |
| [USDA_RUSLE.pdf](USDA_RUSLE.pdf) | Wischmeier & Smith (1978). *Predicting rainfall erosion losses — a guide to conservation planning.* USDA Agriculture Handbook 537 (AH 537) | Method reference | Original nomograph equation, class definitions and validity limits |

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

## 4. Panagos et al. (2014) — Soil erodibility in Europe (LUCAS)

**What it is.** The EU K-factor map (500 m) that GloSEM's K builds on. K was computed with the Wischmeier & Smith nomograph at ~20,000 LUCAS 2009 topsoil points (0–30 cm, one ISO-certified lab), then interpolated with Cubist regression on MODIS, SRTM terrain and latitude/longitude covariates, with Multilevel B-Splines on the residuals (R² = 0.94 fit, 0.74 cross-validation). Mean K = 0.032 t ha h ha⁻¹ MJ⁻¹ mm⁻¹ (SD 0.009, range 0.004–0.076). Mean $K_{st}$ with stone cover = 0.0271.

**Nomograph equation (Eq. 1).** The same as GloSEM, with the SI constant 0.1317:

$$K = \left[\frac{2.1\times10^{-4} M^{1.14}(12-OM) + 3.25(s-2) + 2.5(p-3)}{100}\right]\cdot 0.1317,\qquad M = (m_{silt}+m_{vfs})(100-m_c)$$

**Adaptations to the nomograph** (all relevant to us):
- **Silt cap.** The equation is valid for silt ≤ 70%. Samples with 70–80% silt (718, mostly loess) were set to 70%. Samples above 80% (212) were excluded.
- **Very fine sand.** Not measured, so vfs (0.05–0.1 mm) = **20% of total sand** (0.05–2 mm).
- **OM cap.** OM is capped at **4%**, as required by the nomograph. The authors note this may underestimate the protection in organic-rich soils.

**Structure $s$ (Table 1).** Taken from the European Soil Database (ESDB) pedotransfer rules (Van Ranst et al. 1995), which use the FAO soil name and texture. It is **not derived from texture alone**. ESDB class → $s$: good (G) → 1 very fine granular (1–2 mm); normal (N) → 2 fine granular (2–5 mm); poor (P) → 3 medium/coarse granular (5–10 mm); humic or peaty topsoil (H) → 4 blocky, platy or massive (> 10 mm). Most samples were N, so the **dominant $s$ = 2**.

**Permeability $p$ (Table 2).** Texture classes (USDA NSSH 430; Rawls et al. 1982), with saturated hydraulic conductivity ranges:

| $p$ | Texture | $K_{sat}$ (mm h⁻¹) |
|---|---|---|
| 1 fast and very fast | sand | > 61.0 |
| 2 moderate fast | loamy sand, sandy loam | 20.3–61.0 |
| 3 moderate | loam, silty loam | 5.1–20.3 |
| 4 moderate low | sandy clay loam, clay loam | 2.0–5.1 |
| 5 slow | silty clay loam, sandy clay | 1.0–2.0 |
| 6 very slow | silty clay, clay | < 1.0 |

**Coarse fragments calibrate $p$.** Subsurface fragments (> 2 mm) fill pore space and *slow* infiltration. Following Brakensiek et al. (1986), $K_b = K_f(1-R_w)$: $K_f$ is the fine-earth conductivity of the texture class, and $R_w$ is the coarse-fragment fraction. $p$ is then reassigned from $K_b$ with the ranges above. This moved the dominant $p$ from 3 to **4**.

**Surface stone cover (stoniness).** A separate effect: stones on the surface protect against splash (Poesen et al. 1994). $St = e^{-0.04(R_c-10)}$ for surface stone cover $10\% < R_c < 100\%$, with $K_{st} = K\cdot St$. LUCAS surveyors estimated $R_c$ visually in 5 classes; class mid-points 17.5/37.5/75% give St = 0.740/0.332/0.074. It lowers K by 15% on average across the EU, and by 20–42% in PT, ES, CY, MT, EL and FR.

**Sensitivities reported** (relative to K = 0.032):
| Change | Effect on mean K |
|---|---|
| Ignore coarse fragments in $p$ (dominant $p$ 3 instead of 4) | −15% |
| Ignore structure ($s$ term dropped) | −2.5% |
| Ignore both $p$ and $s$ (common practice) | −16.3% |
| vfs = 33.3% of sand instead of 20% | +11% |
| OM raised from 3.08% to 4% | −9% |
| Stone cover ($K_{st}$) | −15% (strongly regional) |

**Validation.** Against 21 local, regional and national studies, the mean absolute deviation is 14.3% for K and 18.0% for $K_{st}$.

**Relevance.**
- It defines the $p$ table, the vfs = 20% rule, the silt/OM caps and the stone correction used by GloSEM K.
- **$s$ cannot be reproduced globally the Panagos way** (ESDB is EU-only). GloSEM derives $s$ "from texture" (Borrelli et al. 2017), but the lookup is not in the papers we have. Use either a texture lookup (our assumption) or the constant $s$ = 2; the sensitivity is small (~2.5%).
- **Stone handling differs.** Panagos uses *subsurface* fragments to raise $p$ (raises K) and *surface* stone cover for $St$ (lowers K). SoilGrids only has subsurface `cfvo`. Using `cfvo` as $R_c$ for $St$ is a proxy, and must be documented.

---

## 5. Wischmeier & Smith (1978) — USDA Agriculture Handbook 537

**What it is.** The original USLE handbook, which defines all six factors. RUSLE (Renard et al. 1997, AH 703) later revised R, LS and C but kept the K nomograph.

**K definition.** Soil loss per unit of erosion index on the *unit plot*: 72.6 ft (22.1 m) long, 9% slope, continuous clean-tilled fallow, tilled up and down slope. On it, L = S = C = P = 1 and K = A/EI. Measured K for 23 benchmark US soils ranges from 0.03 (Albia gravelly loam) to 0.69 (Dunkirk silt loam) in US units.

**Nomograph (Eq. 3, Fig. 3).** For soils with < 70% silt + very fine sand:

$$100K = 2.1\,M^{1.14}(10^{-4})(12-a) + 3.25(b-2) + 2.5(c-3)$$

- $M$ = (% silt + % vfs) × (100 − % clay). **vfs (0.05–0.10 mm) is moved from sand to silt**, because it erodes like silt; "sand" in the nomograph is 0.10–2.0 mm.
- $a$ = % organic matter (the nomograph curves cover 0–4%).
- $b$ = structure code: 1 very fine granular, 2 fine granular, 3 medium or coarse granular, 4 blocky, platy or massive.
- $c$ = profile permeability class: 1 rapid, 2 moderate to rapid, 3 moderate, 4 slow to moderate, 5 slow, 6 very slow.
- Above ~70% silt + vfs the relation changes (inflections in the chart), and the equation does not apply.
- Worked example, useful as a unit test: silt + vfs 65%, sand 5%, OM 2.8%, $s$ = 2, $p$ = 4 → K = 0.31 (US units) → 0.041 SI.

**Guidance on inputs.**
- **"Many agricultural soils have both fine granular topsoil and moderate permeability"** ($s$ = 2, $p$ = 3). For these, the first part of the nomograph (texture and OM only) already gives K. This supports $s$ = 2 as a default.
- Texture, OM and structure are those of the **topsoil** (for desurfaced soils, the upper 6 in / 15 cm). Permeability is that of the **whole profile**.
- **Coarse fragments are excluded** from sand/silt/clay percentages. Where substantial, they act as a permanent mulch, applied through the mulch chart (Fig. 6) — the same idea as the Panagos $St$ correction.
- Accuracy: 65% of nomograph K values were within 0.02, and 95% within 0.04, of measured K (range 0.03–0.69). It is less sensitive for high-clay subsoils (Fe/Al oxides) and soils with irreversible aggregates.

**Units.**
- AH 537's metric conversion multiplies nomograph K by **1.292**, giving metric t·h/(t·m·cm). That is a different unit system from the SI t ha h ha⁻¹ MJ⁻¹ mm⁻¹.
- For SI, use **0.1317** (Foster et al. 1981), as Panagos et al. and GloSEM do. Do not mix the two.

**LS (for reference).** $LS = (\lambda/72.6)^m(65.41\sin^2\theta + 4.56\sin\theta + 0.065)$, λ in ft; m = 0.5 (≥ 5% slope), 0.4 (3.5–4.5%), 0.3 (1–3%), 0.2 (< 1%). Derived on 3–18% slopes and ~30–300 ft lengths. This is the parent of the Fadl/Shinde LS (Section 2). RUSLE's McCool S and Desmet & Govers L (Section 1) supersede it.

---

## 6. Yang et al. (2024) — Global K mapping and algorithm applicability

**What it is.** A global K map from SoilGrids 2.0 (2020) using four algorithms: USLE-K (nomograph), RUSLE2-K, EPIC-K and Dg-K. Each is checked against a new **global measured-K database**: 340 studies screened down to 144 sites, 1,260 runoff plots and 20,936 plot-years (Table A.1, mainly China, Australia, Brazil and Europe). For plots without a reported K, K was inverted as $K = A/(R\,L\,S\,C\,P)$, with each factor estimated from the plot description (CSLE LS, Borrelli et al. 2017 C, Wischmeier & Smith P). Antarctica and Greenland are excluded.

**RUSLE2-K (Eqs. 3–10; Renard & Dabney 2013).** It keeps the nomograph structure but changes the texture term and the vfs estimate:

$$K = 0.1317\,\frac{K_t K_o + K_s + K_p}{100}$$

$$K_t = \begin{cases}K_{tb} & M_{silt}+M_{vfs}\le 68\%\\ K_{tb} - 0.67\,(K_{tb}-K_{t68})^{0.82} & M_{silt}+M_{vfs} > 68\%\end{cases}$$

$$K_{tb} = 2.1\times10^{-4}\left[(M_{silt}+M_{vfs})(100-M_{clay})\right]^{1.14},\qquad K_{t68} = 2.1\times10^{-4}\left[68\,(100-M_{clay})\right]^{1.14}$$

$$M_{vfs} = (0.74 - 0.62\,M_{sand}/100)\,M_{sand},\qquad K_o = 12 - OM,\qquad K_p = 2.5\,(p-3)$$

- **Below 68% silt + vfs it is identical to USLE-K.** Above 68%, the texture term flattens instead of the silt being capped at 70%.
- **vfs** is a function of sand, not a fixed 20%: it is about 74% of sand in fine soils, 43% at 50% sand and 12% in pure sand. In loamy soils this gives a higher vfs than Panagos, and so a higher K.
- ⚠️ **Structure sign.** The paper prints $K_s = 3.25(2-s)$, the opposite sign to the nomograph's $3.25(s-2)$. That would make blocky/massive soils *less* erodible, which contradicts AH 537. We treat it as a typo and use $(s-2)$.
- OM is capped at 4%, as for USLE-K. OM = SOC (‰) × 0.1724.

**Inputs and preprocessing.**
- **Structure $s$:** from a **WRB soil-type map**. A WRB-to-$s$ table was built from the soil-profile descriptions in the FAO-Unesco (1974) Soil Map of the World manual and applied with a pedotransfer function. ⚠️ **The table is not published.**
- **Permeability $p$:** from texture (Table 3, USDA-NRCS 1983). It matches Panagos Table 2, except that it assigns **silt → 1**. That is implausible (silt has moderate permeability, 3) and probably a typo; we keep 3.
- vfs = 20% of sand for USLE-K (Panagos), and the RUSLE2 equation for RUSLE2-K. Silt is capped at 70% for USLE-K. No coarse-fragment or stone correction is applied.

**Results (at the 144 sites).**
| | Measured | USLE-K | RUSLE2-K | EPIC-K | Dg-K |
|---|---|---|---|---|---|
| Mean K | 0.0243 | 0.0379 | **0.0282** | 0.0397 | 0.0399 |
| Range | 0.0001–0.0674 | 0.0219–0.0598 | 0.0131–0.0489 | 0.0190–0.0548 | 0.0112–0.0438 |
| Sites overestimated | — | 86.8% | 73.6% | 85.4% | 79.1% |

All four algorithms overestimate measured K. RUSLE2-K is closest. Silt content is the most important driver of K, followed by OM, clay, structure and sand.

**Correction equations (estimated vs measured K, Eqs. 25–28).**
| Algorithm | Correction | R² | NSE |
|---|---|---|---|
| USLE-K | $2.0769K - 0.0543$ | 0.611 | 0.61 |
| **RUSLE2-K** | $2.1311K - 0.0359$ | **0.636** | **0.64** |
| EPIC-K | $1.8722K - 0.05$ | 0.414 | 0.41 |
| Dg-K | $0.9559K - 0.011$ | 0.106 | 0.11 |

Only the corrected USLE-K and RUSLE2-K meet their acceptance criteria (R² > 0.5, NSE > 0.4). The corrected RUSLE2-K map agrees in pattern with Borrelli et al. (2017), Panagos et al. (2014) and regional maps (Greece, eastern China, NSW, Brazil).

**Caveats for our use.**
- **The correction is a steep linear rescale** (slope 2.13) fitted on 144 sites that cluster in four regions, with many K values back-calculated from assumed R, LS, C and P. It goes **negative for K < 0.0168**, so it must be floored. In our 0–5 cm dry run, 2.1% of pixels hit the floor, and the spread widens (p99 0.048 → 0.066). Use it as a sensitivity or alternative layer, not as a silent default.
- **Uncorrected RUSLE2-K stays comparable with GloSEM / Panagos K** (same nomograph family; our dry-run mean is 0.033 vs EU 0.032).
- The EPIC-K correction is poor (R² 0.41), which weakens EPIC as our QA layer. It is still useful as a texture-only cross-check.

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
