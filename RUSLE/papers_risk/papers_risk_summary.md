# Climate-risk and hazard papers — summary

These documents inform how to make the RUSLE **R layer scenario- and hazard-aware**: precipitation scenarios, drought and flood/extreme-rain years, and their probabilities. They also show how to turn erosion into risk metrics (expected annual and extreme-year values) and into yield or financial impacts.

| File | Reference | Type | Key contribution for us |
|---|---|---|---|
| [EAFRD_AGRI_Insurance_Risk_MA.pdf](EAFRD_AGRI_Insurance_Risk_MA.pdf) | fi-compass / Howden, Risklayer & KIT (2025). *Insurance and Risk Management Tools for Agriculture in the EU — Market analysis.* EIB/EC. | Report (~190 pp.) | Full catastrophe-modelling recipe: hazard event sets (10,000 simulated years), exceedance probabilities, GPD tails, CMIP6 climate conditioning |
| [04. Insurance and risk management tools …fi-compa.pdf](<04. Insurance and risk management tools for agriculture in the EU – Findings of the fi-compa.pdf>) | González Peláez, Daniell & Douglas (2025). Slide deck summarising the study above. | Presentation | Headline numbers; risk-layering concept |
| [approach-and-methodology-for-climate-risk-assessments.pdf](approach-and-methodology-for-climate-risk-assessments.pdf) | Alliance of Bioversity–CIAT. *Approach and methodology for climate risk assessments — Supply shed model.* | Method note | Standard hazard index definitions (drought = SPEI, extreme rain = Rx5day), baseline and future windows, Do Nothing / Adapt scenarios |
| [Assessing the impact of extreme climate events on European gross .pdf](<Assessing the impact of extreme climate events on European gross .pdf>) | Zhang, Loaiciga, Okujeni, Liu, Tan & Sauter (2025). Agric. For. Meteorol. 362:110374. [doi:10.1016/j.agrformet.2024.110374](https://doi.org/10.1016/j.agrformet.2024.110374) | Journal article | Catalogue of ETCCDI extreme-precipitation and drought indices at 10 km; how each affects vegetation (and therefore cover / C) |
| [FOLU-Re-Wire-2026-Report.pdf](FOLU-Re-Wire-2026-Report.pdf) | FOLU / Systemiq / WBCSD (2026). *Re-WIRE — Agricultural Resilience: From cost to investment opportunity.* | Report | Framing: "probabilistic risk-to-resilience modelling" (hazard × exposure × vulnerability → AAL/PML → return on investment of resilience) |
| [hadi et al 2026.pdf](<hadi et al 2026.pdf>) | Hadi, Ray, Borrelli, Gerber, Shokri, Roman & Wuepper (2026). *Land degradation is associated with larger crop yield gaps across global croplands.* Nature Food 7:678–687. [doi:10.1038/s43016-026-01382-5](https://doi.org/10.1038/s43016-026-01382-5) | Journal article | Empirical link from erosion to yield gap: elasticity 0.044, at 10 km, the same grid scale as ours |

---

## 1. fi-compass market analysis (EAFRD) and slide deck

**Scope.** First EU-wide, insurance-grade quantification of climate losses to agriculture (EU-27, crops and livestock), now and in 2050. It covers four primary perils (**drought, excess rain, hail, frost**) and assesses the protection gap in 10 case-study countries.

**Headline numbers.**

- EU-27 Annual Average Loss (AAL) is about EUR 28.3 billion, ~6% of production (crops EUR 17.4 billion, livestock EUR 10.9 billion). It rises to over EUR 40 billion by 2050.
- The 1-in-50-year Probable Maximum Loss (PML, 2% annual probability) is about EUR 60 billion now and over EUR 90 billion in 2050.
- Only 20–30% of losses are insured.
- A catastrophe year costs Member States 4–5 times the AAL, but only about 2 times at EU level, thanks to diversification.
- Drought dominates in most Member States.

**Method (Part 1.1 and Annex II). Directly reusable for a stochastic R/erosion model:**

1. **Hazard module.** Combines historical hazard data, stochastic statistics and climate models. It simulates **10,000 synthetic years at current risk levels**, not a replay of history. A 1-in-100-year estimate needs many 100-year blocks, which is why 1,000 years is not enough.
2. **Hazard indicators from ERA5 (~25 km, 1960–2023).** Each peril is reduced to one value per year within a seasonal window (March–September for rain and drought):
   - strong precipitation days (≥ 10 mm/day);
   - **maximum 5-day precipitation (Rx5day)**;
   - topsoil soil-moisture anomaly against 1960–1990;
   - SPEI and PDSI event sets for drought;
   - frost metrics using growing degree days (GDD).
3. **Area-percentile statistics.** Hazard intensity is computed for the most-affected 1%, 5%, 10% and 33% of crop area in each country, weighted with CROPGRIDS. This captures the fact that extremes are spatially limited.
4. **Probability model.**
   - Annual values are fitted with a **normal distribution**: $P(X>T) = 1-\Phi((T-\mu)/\sigma)$.
   - Extreme continuous precipitation uses a **lognormal** distribution: $1-\Phi((\ln T-\mu)/\sigma)$.
   - Zero-inflated variables use a two-part form: $p_0\,[1-\Phi((T-\mu_+)/\sigma_+)]$.
   - Period shifts are compared through the mean, e.g. 1970–1990 against 2000–2020.
5. **Tails.** Peaks-over-Threshold with **Generalised Pareto Distributions** extends loss curves to 200-year return periods.
6. **Climate conditioning.** A bias-corrected ensemble of **16 CMIP6 models** (SSP2-4.5, SSP5-8.5; CIL GDPCIR downscaling at 0.25°), using the ensemble mean. The inter-model SD is about ±13%; the lowest model is about −22% and the highest about +27%. Event frequencies are adjusted by the return-period change in the stochastic set.
   - Example: Rx5day intensity rose by ~3% between 1980 and 2010 and rises by up to 15% by 2050; EU soil moisture has fallen ~10% since 1980.
7. **Vulnerability.** Trained on 40 years of yield and insurance losses (Olympic Yield Index method).
8. **Metrics.**
   - **AAL**: mean over all simulated years. AALs can be summed across perils and regions.
   - **PML**: loss at a return period. PMLs **cannot** be summed.
   - Cross-country correlation matrices handle EU-wide aggregation.
9. **Risk layering.** Attritional losses (high frequency, low severity) are retained by farmers; severe losses go to insurance; catastrophic losses fall to the public sector, reinsurance or cat bonds.

**Relevance.** This provides the blueprint for the Monte Carlo R / erosion module:

- the event-set size;
- the distributions per variable type;
- Rx5day as the excess-rain hazard and SPEI/soil moisture as the drought hazard;
- CMIP6 conditioning;
- reporting as mean annual erosion (an AAL analogue) and 1-in-N-year erosion (a PML analogue).

---

## 2. Bioversity–CIAT — Approach and methodology for climate risk assessments (supply shed model)

**Scope.** Method behind CIAT's Climate Resilience Platform. It projects crop-yield impacts of climate hazards and the benefit of adaptation practices across crops and countries (canola, corn, oats, rice, soybean, sugar beet, sugarcane, sunflower, wheat …).

**Key elements.**

- **Hazard indices over the crop-specific growing season.** The growing season comes from crop calendars (USDA IPAD, FAO GIEWS, Sacks et al. 2010).
  - Heat: days above the crop thermal limit.
  - Frost: days below 0 °C.
  - **Extreme rainfall: maximum 5-day accumulated precipitation.**
  - **Drought: SPEI**, calibrated on 1981–2010.
- **Time windows.**
  - Baseline: 1980–2010.
  - Future: **21-year windows centred on the target year** (2020–2040 for "2030", 2030–2050 for "2040").
- **Climate data.** Daily CMIP6 with ERA5-Land, WorldClim and TerraClimate as references. GCM subsets are chosen per region for availability and consistency.
- **Thresholds.** Crop-specific hazard thresholds from a systematic literature review (Annex C/D). Yield impacts fall into 6 classes, from "Much higher yield" to "Extreme yield loss".
- **Scenarios.**
  - *Do Nothing*: future climate with current practices.
  - *Adapt*: practices scored by impact potential (Low 10%, Medium 40%, High 70% risk reduction × yield-benefit multipliers 0.4 / 0.8 / 1.0) and by incremental adoption (target minus current).
  - Production areas from CROPGRIDS.

**Relevance.**

- Standard, defensible definitions for the **drought (SPEI) and flood/extreme-rain (Rx5day) hazards**, and for baseline and future windows.
- The *Do Nothing / Adapt* framing maps directly onto LEAF management options. Conventional tillage and residue removal correspond to Do Nothing; reduced tillage, residues and cover crops correspond to Adapt via $C_{mgmt}$.

---

## 3. Zhang et al. (2025) — Extreme climate events and European GPP

**Scope.** Explains the end-of-growing-season gross primary production (MODIS GPP) over Europe (2003–2020, 10 km grid, 16-day steps) as a function of extreme-climate and drought indices, using an explainable LightGBM model with SHAP values. It separates rainfed cropland, irrigated cropland and grassland, and cold, temperate and arid climates.

**Indices (ETCCDI and drought).**

| Type | Indices |
|---|---|
| Precipitation | Rx1day, **Rx5day**, **R95p** (very wet days), CT (consecutive-day total), CDD (dry spell length), CWD (wet spell length) |
| Temperature | TXx, TNx, TX90p, TN90p |
| Drought | SMA (0–5 cm soil moisture), SMI (0–50 cm root-zone soil moisture), SWDI, **SPI**, **SPEI**, CDI (combined) |

Data: E-OBS 0.1° daily, SoMo.ml-EU, GSSM1km, MODIS ET/PET, SoilGrids FC/WP, GFSAD irrigated/rainfed map, Köppen–Geiger.

**Findings.**

- Early-season soil moisture and absolute temperature extremes are the strongest predictors ("drought memory").
- In arid climates, extreme precipitation amounts dominate.
- In cold climates, agricultural drought dominates.
- Rainfed cropland and grassland respond more strongly than irrigated cropland.
- Rx1day and R95p are the best precipitation predictors.

**Relevance.**

- **R95p / Rx1day / Rx5day** are the natural hazard covariates for **extreme-rain (flood) years**, because erosivity is concentrated in intense events.
- Drought indices drive **vegetation loss**, which in RUSLE appears as a higher **C factor**, not a higher R. A drought followed by intense rain is the high-erosion compound case.
- Irrigation buffers the effect, which is consistent with the rainfed/irrigated split in our land uses.

---

## 4. FOLU Re-WIRE (2026) — Agricultural resilience: from cost to investment opportunity

**Scope.** Phase 2 of FOLU's Re-WIRE programme. It focuses on oilseeds (soy, palm, rapeseed, sunflower), starting with the Brazil–China soy corridor.

**Key messages.**

- Physical risks are **larger and more correlated** than markets assume. Simultaneous droughts in the US, Argentina and Brazil (~80% of global soy) become 11–16 times more likely and 4–15% more severe under long-term trends.
- A shock in one country raises soybean prices by 1.8–5.9%; all three together raise them by 9.5–33.2%.
- Current corporate assessments stop at climate or sectoral impact models. They should adopt **catastrophe risk modelling** (hazard × exposure × vulnerability → AAL and PML) and extend it to **probabilistic risk-to-resilience modelling**: AAL or PML avoided by an intervention (no-till, cover crops, watershed restoration), compared with its cost to give an ROI.
- Economy-wide returns on agricultural adaptation are about 27%.
- Evidence gaps include compound events and nature-related risks, **including soil erosion** and pollinator loss.

**Relevance.**

- Positions erosion explicitly as a nature-related risk that belongs in the same probabilistic framework.
- Supports reporting the stochastic erosion output as AAL/PML-type metrics, so that management options can be valued as avoided erosion and avoided loss.

---

## 5. Hadi et al. (2026) — Land degradation and crop yield gaps (Nature Food)

**Scope.**

- Global cross-sectional analysis of the association between five land-degradation "debts" (current minus historical condition) and crop-yield gaps for 10 crops, circa 2010, at **5 arc-min (10 km), n = 405,084 cells**.
- The five debts: soil erosion rate, compaction, SOC, soil water, tree cover. The erosion debt comes from the Borrelli group.
- Method: log–log multiple regression with 41 controls (climate, soil, terrain, inputs, socio-economics), plus a **causal forest** for local heterogeneity.

**Findings.**

- A 10% increase in overall degradation goes with a ~1.75% (CI 1.69–1.82%) larger yield gap.
- **For soil erosion debt alone: 0.44% per 10% (elasticity ≈ 0.044, CI 0.040–0.049).**
- Local elasticities reach 4.4–22.7% per 10% in hotspots: northern and southern India, north-eastern China, the US Midwest, northern Argentina.
- Global losses per 1% increase in degradation: about 2.0 Mt crops, USD 433 million, and calories for about 7.1 million people per year.
- Highest relative country losses: Lesotho, Botswana, Burundi.
- Intensive, high-yield systems are the most sensitive.

**Relevance.**

- Provides a **vulnerability function** from erosion to yield on the same 10 km grid as our LEAFs. A scenario- or hazard-driven change in erosion ($\Delta A/A$) can be translated into a yield-gap change ≈ 0.044 × %ΔA, or a local causal-forest elasticity where available.
- Together with crop prices, this closes the hazard → erosion → yield → value chain described in the FOLU and fi-compass reports.

---

## Synthesis — what these documents imply for the RUSLE R layer

1. **Hazards to model.**
   - *Excess rain / flood*: Rx5day, R95p or Rx1day, which raise R.
   - *Drought*: SPEI or soil-moisture anomaly. It mainly lowers vegetation cover, so C rises; R falls slightly in drought years, and erosion rises when rain returns.
2. **Probability machinery.**
   - Thousands of simulated years (10,000 in fi-compass).
   - Normal or lognormal annual distributions, and GPD tails for return periods beyond the record.
   - Spatial correlation preserved by resampling whole years or using correlation matrices.
3. **Climate conditioning.**
   - A CMIP6 multi-model ensemble (SSP1-2.6 / 2-4.5 / 5-8.5).
   - A 1980/1981–2010 baseline and ~20-year future windows centred on 2030, 2050 and 2070.
   - Keep the inter-model spread as an uncertainty term (about ±13–27%).
4. **Outputs.**
   - Mean annual soil loss (AAL analogue).
   - 1-in-10 and 1-in-50-year soil loss (PML analogues).
   - Probability of exceeding a tolerance threshold (e.g. > 10 t ha⁻¹ yr⁻¹).
5. **Impact link (optional).** Erosion change → yield gap (Hadi et al. elasticity) → production and financial loss (FOLU / fi-compass framing). Management options act as *Adapt* scenarios.
