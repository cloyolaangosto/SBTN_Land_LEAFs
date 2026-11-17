# `code_py/`: Python port of the Hadi et al. model scripts

This folder translates the two model-fitting scripts of the replication package for
Hadi et al., *"Land degradation is associated with larger crop yield gaps across
global croplands"* (Nature Food, forthcoming) from R to Python:

| R original | Python port |
|---|---|
| `code/02_models/01_fit_regression_models.R` | `code_py/02_models/01_fit_regression_models.py` |
| `code/02_models/02_fit_causal_forests.R` | `code_py/02_models/02_fit_causal_forests.py` |
| `code/_shared/paths.R`, `analysis_paths.R` | `code_py/_shared/paths.py` |
| `code/01_data/dataset_contract.R` (+ variable lists inlined in the R scripts) | `code_py/_shared/dataset_contract.py` |
| `fixest::conley()`, `marginaleffects::avg_slopes()` | `code_py/_shared/econometrics.py` |

The rest of this document covers four things:

1. what the model folder does and where it sits in the pipeline
2. what each script does, step by step
3. the theory behind each method
4. how each piece was translated, and where the Python results will differ from R

---

## 1. Where the model scripts sit

```
01_data/  (GEE export -> grid table -> analysis-ready .rds)
   │
   ▼
02_models/
   01_fit_regression_models.R    ← log-log OLS: average association + robustness checks
   02_fit_causal_forests.R       ← causal forests: per-cell heterogeneous effects (CATEs)
   03_create_cate_inspection_outputs.R
   04_compute_loss_rasters_and_tables.R   (uses the CATE rasters -> production/calorie/revenue losses)
   │
   ▼
03_figures/, 04_tables/
```

The two scripts answer two different questions:

* **01: what is the average association?** Across all global cropland cells, by
  how many percent does the yield gap rise when a degradation "debt" rises by 1 %,
  holding climate, terrain, soil, management and socioeconomics fixed? This
  feeds Figs. 2–3 and Supp. Figs. 3, 5 and 6.
* **02: where is it larger or smaller?** For each 10 km cell, what is the local
  effect of each debt on the yield gap? These CATE maps feed Supp. Figs. 8–14
  and the loss calculations in `04_compute_loss_rasters_and_tables.R` (Figs. 5–6).

### 1.1 Unit of analysis and variables

Each row is one cropland grid cell of about 10 km, with `long`/`lat` at the cell centre.
The table is built in `01_data/02_build_stage2_grid_table.R` and kept only where
the 2010 yield gap is observed.

**Outcome.** `AllCrops2010YieldGapPercent` is the aggregate yield gap in 2010,
as a percentage of attainable yield. Two secondary outcomes are also used:
`actualYieldGerberRay` and `attainableYieldGerberRay`.

**Treatments: five land-degradation "debts".** Each is a change relative to an
earlier reference state. Negative values (improvements) are set to 0
(`pmax(x, 0)`), so a debt measures only deterioration:

| Treatment | Source layer | Meaning |
|---|---|---|
| `soilErosionDebt` | `Diff_Water_erosion__mean` | increase in water erosion |
| `soilCompactionDebt` | `Subsoil_compaction_susceptibility_index_mean__mean` | subsoil compaction susceptibility |
| `soilOrganicCarbonDebt` | `soc_delta_ca1983_min_ca2010__mean` | SOC lost between ~1983 and ~2010 |
| `soilWaterDebt` | `sm_1979_1983_mean_minus_2008_2012_mean__mean` | soil-moisture decline, 1979–83 → 2008–12 |
| `treeCoverDebt` | `Diff_treeCover__sum` | tree-cover loss |

**Controls**, in the groups the scripts use:

| Group | Contents |
|---|---|
| natural: climate (WorldClim) | srad, tavg, prec, warmest-quarter temperature, max temperature of warmest month, wettest-quarter precipitation, aridity |
| natural: climate (TerraClimate) | end-of-period precipitation, tmin, tmax, srad, plus their **deltas** over the period |
| natural: terrain | elevation, slope, terrain ruggedness (TRI) |
| natural: soil | dominant soil type (a categorical *factor*), SoilGrids clay/sand/silt |
| adaptation | N fertilizer (NH4, NO3), manure, N deposition (NOy, NHx), Olsen P, pesticides, irrigated area |
| socioeconomic | HDI, GDP (PPP), GDP per capita, property rights, enforcement, rural electricity, phone subscriptions, agricultural GDP share, agricultural employment, corruption, machinery |

**The `__rescaled__log` transform.** For every numeric variable, the data
pipeline also creates

```
x__rescaled       = (x - min(x)) / (max(x) - min(x)) * 100 + 1      ∈ [1, 101]
x__rescaled__log  = log(x__rescaled)                                ∈ [0, log 101]
```

Rescaling to [1, 101] means the log is always defined, including for debts that
are exactly 0 (which is common after clipping). The regressions use only
`__rescaled__log` variables, so they are log-log models. The causal forests use
the raw variables, because trees are invariant to monotone transforms of X
(only the scale of Y and W matters for the CATE units).

> Caveat: in `01_data/03_build_final_analysis_table.R`, actual and attainable yield
> are rescaled with a different function, `x - min(x) + 1`, with no division by the
> range. Their logs are still well defined, but they are on a different scale from
> the other `__rescaled__log` variables.

---

## 2. `01_fit_regression_models.R`: log-log OLS

### 2.1 What the script does

1. It defines the outcome, the 5 treatments and the 7 control groups, all on the log scale.
2. It builds the **baseline formula** `f`:
   `log YG ~ 5 log debts + natural + adaptation + socioeconomic controls`.
3. It reads the analysis-ready `.rds` and converts the crop-type dummies to factors.
4. It saves the variable lists (`model_spec_variables.rds`).
5. It fits each specification with `fixest::feols`, which here is plain OLS because
   there are no fixed effects. It saves the fitted objects into three folders
   that differ only in the variance estimator:

   | folder | `vcov` | meaning |
   |---|---|---|
   | `models/main_10km` | `NULL` | the fixest default. With no fixed effects this is **iid** (classical) SEs |
   | `models/hc1` | `"HC1"` | heteroskedasticity-robust (baseline only) |
   | `models/conley_100km` | `conley(cutoff = 100)` | Conley spatial-HAC SEs, 100 km cutoff |

6. It fits these specifications:

   | Model file | Change from baseline | Purpose |
   |---|---|---|
   | `model_baseline` | — | headline estimate |
   | `model_winsorized` | all numeric vars capped at their 2.5th / 97.5th percentiles | outliers |
   | `model_no_controls` → `model_natural_controls` → `model_natural_adaptation_controls` → `model_full_controls` | the control ladder (Fig. 2) | sensitivity to control sets |
   | `model_long_lat_polynomial` | + long, lat, long², lat², long·lat | smooth spatial trends |
   | `model_without_soilgrid_controls` | − clay/sand/silt | collinearity with the soil debts |
   | `model_yield_gap_crop_type_controls` (also saved as `model_crop_type_controls`) | + 10 dominant-crop dummies | crop composition |
   | `model_field_size_control`, `_soil_depth_`, `_soil_n_`, `_water_table_depth_`, `_road_density_`, `_travel_time_to_cities_`, `_agricultural_expenditure_`, `_economic_freedom_` | + one extra covariate | omitted-variable checks (Supp. Fig. 3) |
   | `model_without_machinery_control` | − machinery | possible bad control |
   | `model_actual_yield`, `model_attainable_yield` (± crop dummies) | outcome swapped | decomposes the gap (Fig. 3, Supp. Fig. 6) |
   | *(opt-in, `LDYG_ALLOW_AVG_SLOPES`)* `model_quadratic_land_degradation` | + the square of each debt | nonlinearity |
   | `model_interaction_land_degradation` | + all 26 products of 2–5 debts | joint degradation |
   | `model_interaction_land_degradation_adaptation` | + the above + the 40 debt × adaptation products | does management buffer degradation? |

   For the three nonlinear models it also saves `marginaleffects::avg_slopes(...)`,
   the average marginal effect of each debt, so they can be compared with the
   linear coefficients in Fig. 2.

(`coarser_dir` is defined but unused in this script. The coarser-resolution
inputs are produced elsewhere.)

### 2.2 Theory

**Model.** For cell *i*:

$$\log YG_i = \alpha + \sum_{k=1}^{5}\beta_k \log D_{ik} + \gamma' Z_i + \varepsilon_i$$

Both sides are logged, so each $\beta_k$ is an **elasticity**: a 1 % increase in
(rescaled) debt *k* is associated with a $\beta_k$ % change in the (rescaled) yield
gap, holding the other debts and all controls $Z_i$ fixed. Because all five debts
enter together, each $\beta_k$ is a partial effect net of the co-occurring
degradation processes.

**Identification: selection on observables.** The estimate is causal only if,
conditional on $Z$, the debts are as good as randomly assigned
($E[\varepsilon \mid D, Z] = 0$). The controls are meant to absorb the main
confounders: climate and climate change, terrain, inherent soil properties,
management intensity, and institutions and wealth. The title of the paper says
"associated with", which reflects this assumption.

**The control ladder and "bad controls".** Adaptation and socioeconomic variables
are partly *outcomes* of degradation. For example, farmers may add fertilizer or
irrigation on degraded land. Conditioning on such mediators can absorb part of the
effect, or even induce bias. Fitting no → natural → natural+adaptation → full
controls shows how much the estimate depends on that choice. Stable coefficients
across the ladder are reassuring.

**Conley standard errors.** Neighbouring 10 km cells share unobserved weather,
soils and management, so residuals are spatially correlated. iid or HC1 SEs then
understate uncertainty. Conley (1999) allows arbitrary correlation between any two
cells within a cutoff distance:

$$\hat V = (X'X)^{-1}\Big[\sum_i\sum_j K(d_{ij})\,x_i\hat u_i\,\hat u_j x_j'\Big](X'X)^{-1},\qquad K(d)=\mathbb 1[d\le 100\text{ km}]$$

**Nonlinear terms and average slopes.** With $\beta_1\log D_1 + \beta_{11}(\log D_1)^2 + \beta_{12}\log D_1\log D_2 + \dots$,
the effect of $D_1$ varies across cells. `avg_slopes` reports the **average
marginal effect**,

$$\text{AME}_1 = \frac1n\sum_i \frac{\partial \widehat{\log YG}_i}{\partial \log D_{i1}} = \beta_1 + 2\beta_{11}\overline{\log D_1} + \beta_{12}\overline{\log D_2}+\dots,$$

which is directly comparable to $\beta_1$ in the linear model. Its standard error
comes from the delta method.

**Winsorizing** at 2.5/97.5 % tests whether extreme cells drive the estimate.
The **long/lat polynomial** absorbs smooth large-scale spatial gradients that
might correlate with both degradation and yields.

**Actual vs attainable yield.** The yield gap is roughly
(attainable − actual)/attainable. Re-estimating with each component as the outcome
shows whether degradation widens the gap by lowering actual yields or by raising
the attainable benchmark.

### 2.3 Python translation

| R | Python |
|---|---|
| `as.formula(paste(...))`, `update(f, . ~ . + x)`, `update(f, y2 ~ .)` | the small `Spec` dataclass with `.add()`, `.drop()`, `.with_outcome()`, `.formula()` |
| `make_formula(outcome, controls)` | `make_formula(outcome, controls)` |
| `fixest::feols(formula, data, vcov = NULL / "HC1")` | `pyfixest.feols(formula, data, vcov="iid" / "HC1")` |
| `vcov = conley(cutoff = 100)` | fit with iid, then `econometrics.conley_vcov(X, resid, lat, long, 100)` |
| `I(x^2)` | `I(x ** 2)` |
| `a:b:c` (numeric product) | `a:b:c` (same syntax) |
| `combn(TREATMENTS, i, paste, collapse=":")` | `itertools.combinations` |
| `winsorize()` with `quantile(type 7)` | `winsorize()` with `Series.quantile` (linear, identical to type 7) |
| `marginaleffects::avg_slopes(m, variables = TREATMENTS)` | `econometrics.avg_slopes(fit, data, TREATMENTS)` |
| `write_rds(model, "x.rds")` | `x.pkl` (a pickled `FitResult`: coefs, vcov, n, R², formula) + `x_coefs.csv` (tidy table) |
| `allow_heavy("LDYG_ALLOW_AVG_SLOPES")` | `allow_heavy("LDYG_ALLOW_AVG_SLOPES")` |

Implementation notes:

* **Conley** (`conley_vcov`). Neighbours within 100 km come from a haversine
  `sklearn` `BallTree`. The "meat" is built in chunks with a sparse
  neighbour matrix ($\sum_{j\in N(i)} s_j$ for each $i$), so the N × N matrix
  is never held in memory. It uses a uniform kernel and the fixest small-sample
  factor $(n-1)/(n-k)$. With a global 10 km grid, each cell has about 300
  neighbours, so memory use is about (chunk size × 300) non-zeros per chunk.
* **avg_slopes**. marginaleffects differentiates numerically. The Python version
  differentiates each coefficient term analytically (a plain variable, `I(x ** 2)`,
  and `:` products). It averages the gradient row-wise to get $\bar g$, then computes
  $\text{AME}=\bar g'\hat\beta$ and $SE=\sqrt{\bar g'\hat V\bar g}$. The AME is
  linear in β, so the delta method is exact. The p-values use the normal
  distribution, as marginaleffects does.
* **NA handling**. fixest drops incomplete rows model by model. The Python version
  does the same explicitly in `estimation_sample()`, so the rows passed to
  `avg_slopes` and Conley line up with the fit.
* **Outputs** go to `data/processed/models_py/{main_10km,hc1,conley_100km}/`, which
  keeps them separate from the R `.rds` files.

---

## 3. `02_fit_causal_forests.R`: causal forests and CATE maps

### 3.1 What the script does

**Stage A: fit.** This stage is opt-in via `LDYG_ALLOW_CAUSAL_FOREST_FIT=true`. In R
it needs about 8 h and 20–30 GB RAM per forest.

1. It reads the analysis-ready data and keeps the **raw** (unlogged) outcome,
   treatments, controls and crop dummies. It drops every row with any NA
   (`tidyr::drop_na`) and saves this sample as `analysis_ready_data_linear_scale.rds`.
2. It builds the covariate matrix `X_base`: all numeric controls, plus a one-hot
   encoding of soil type (`model.matrix(~ soil_type__mode - 1)`, one column per
   level), plus the 10 crop-type dummies.
3. `run_cf_func_customXbase(D_var)` runs once for each of the five debts:
   * `W` = that debt (a **continuous** treatment), `Y` = yield gap %
   * `X` = `X_base` + **the other four debts**
   * `grf::causal_forest(X, Y, W, num.trees = 2000, min.node.size = 5,
     tune.parameters = "none", imbalance.penalty = 0, seed = 42)`
   * `predict(cf)` with no new data gives **out-of-bag** CATE predictions
   * it saves the forest (`causal_forest_<process>_crop_type_controls.rds`) and
     a `long, lat, CATE` table (`cate_<process>_crop_type_controls.{csv,rds}`)

**Stage B: rasterize.** This stage is opt-in via `LDYG_ALLOW_CATE_RASTERIZE=true`.

1. It inner-joins the five CATE tables on `long, lat`.
2. It rasterizes each CATE column onto the grid of `raw/crop_yield/yield_gap_2010_percent.tif`
   (`raster::rasterize(fun = "max")`).
3. It sets **negative CATEs to 0** and keeps NA cells as NA.
4. It writes `cate_{erosion,compaction,soc,soil_water,tree_cover}_nonneg.tif`, plus
   `cate_sum_nonneg.tif`, the cell-wise sum of the five layers. All are
   Float32, DEFLATE-compressed, with 512 × 512 tiles.

### 3.2 Theory

**Estimand.** Write the outcome as a partially linear model with a heterogeneous slope:

$$Y_i = \tau(X_i)\,W_i + m(X_i) + \varepsilon_i .$$

$\tau(x)$ is the **conditional average treatment effect**: the expected change in
yield-gap percentage points per one-unit increase in the debt (in raw units)
for a cell with characteristics $x$. With a continuous $W$, it is a local
average slope.

**Robinson / R-learner residualization.** Subtracting the conditional means
removes $m(X)$:

$$Y_i - \hat m(X_i) = \tau(X_i)\,(W_i - \hat e(X_i)) + \varepsilon_i,\qquad \hat m = \hat E[Y\mid X],\ \hat e = \hat E[W\mid X].$$

grf estimates $\hat m$ and $\hat e$ with regression forests and uses their
**out-of-bag** predictions ("local centering"). This makes the
$\tau$ estimate robust to first-order errors in the nuisance functions
(Neyman orthogonality).

**Generalized random forest** (Athey, Tibshirani & Wager, 2019). For a target
point $x$, the forest defines adaptive neighbourhood weights $\alpha_i(x)$: how
often cell $i$ lands in the same leaf as $x$. It then solves the local moment
condition

$$\sum_i \alpha_i(x)\,\big(\tilde Y_i - \tau(x)\tilde W_i - b(x)\big)\binom{\tilde W_i}{1} = 0,$$

which is a weighted local regression of centered Y on centered W. Two design
choices make the result a valid estimator:

* **Gradient-based splitting.** Trees split to maximize heterogeneity in
  $\tau$, not to predict Y.
* **Honesty.** Each tree uses half of its subsample to choose splits and the
  other half to estimate leaf values. This gives asymptotically normal,
  consistent CATEs.

**Why the other debts go into X.** Each forest's CATE for debt *k* holds the
other four debts fixed, as the joint OLS does, and it lets the effect of *k*
vary with the other debts (for example, erosion may matter more where SOC is
already depleted).

**Clipping negative CATEs to 0 before summing.** The maps and the loss accounting
in `04_compute_loss_rasters_and_tables.R` focus on cells where degradation is
associated with *larger* yield gaps. The sum of the five non-negative layers is an
aggregate "degradation-attributable yield-gap" surface. This is a deliberate
modelling choice: it rules out offsetting negative effects, so the aggregate
should be read as an upper-leaning estimate.

### 3.3 Python translation

| R (`grf`) | Python (`econml.grf`) |
|---|---|
| `regression_forest` for `Y.hat`, `W.hat` (inside `causal_forest`) | `RegressionForest(honest=True)` × 2, with `.oob_predict(X)` |
| `causal_forest(X, Y, W, ...)` | `CausalForest(criterion="het", fit_intercept=True, honest=True).fit(X, W − Ŵ, Y − Ŷ)` |
| `predict(cf)$predictions` (OOB) | `cf.oob_predict(X)` |
| `num.trees = 2000` | `n_estimators = 2000` (nuisance forests: `max(50, 2000/4)` = 500, as in grf) |
| `min.node.size = 5` | `min_samples_leaf = 5` |
| `sample.fraction = 0.5` (default) | `max_samples = 0.5` |
| `mtry = min(ceil(sqrt(p) + 20), p)` (default) | `max_features = grf_mtry(p)` |
| `alpha = 0.05` (default) | `min_balancedness_tol = 0.45` (each child keeps ≥ 5 % of its parent) |
| `ci.group.size = 2` (default) | `subforest_size = 4` |
| `honesty = TRUE` (default) | `honest = True` |
| `set.seed(42)` | `random_state = 42` |
| `model.matrix(~ soil_type__mode - 1)` | `pd.get_dummies(soil_type__mode)` (all levels) |
| `tidyr::drop_na()` | `DataFrame.dropna()` |
| `raster::rasterize(points, template, field, fun = "max")` | `rasterize_points_max`: affine inverse → row/col, then `np.maximum.at` |
| `terra::ifel(r < 0, 0, r)` | `np.where(r < 0, 0, r)` (NaN preserved) |
| `terra::writeRaster(..., COMPRESS=DEFLATE, BLOCK*=512, FLT4S, NAflag=-3.4e38)` | `rasterio` profile with the same options |
| `write_rds` / `fwrite` | `.pkl` (forest + Ŷ, Ŵ) / `.csv` + `.parquet` (CATE tables) |

These overrides exist only in the Python version: `LDYG_CF_NUM_TREES` and
`LDYG_CF_NUM_THREADS`, for quick test runs. Outputs go to
`data/processed/causal_forest_py/`.

---

## 4. Differences from R and caveats

* **The causal forests will not match R exactly.** econml's GRF follows the same
  algorithm but differs in random-number streams, split-search details, and
  grf's `stabilize.splits` (which econml does not reproduce exactly). To
  compare, use the CATE correlation between R and Python, the mean CATE, and the
  maps, not cell-by-cell identity. The OLS coefficients and the iid/HC1 SEs
  should match fixest to numerical precision.
* **Conley distances.** The Python version uses exact haversine distance with a
  6371 km Earth radius. fixest uses its own distance approximation, so pairs right
  at the 100 km boundary may be classified differently and the SEs can differ
  slightly.
* **Factor coefficient names differ.** R gives `dominantCropGerberRay_rice1` and
  `soil_type__modeVertisols`. pyfixest gives `dominantCropGerberRay_rice[T.1]` and
  `soil_type__mode[T.Vertisols]`. The reference level is the first factor level
  in both.
* **Reading `.rds`.** `pyreadr` reads data frames only, so the R fitted-model
  `.rds` files cannot be loaded in Python. R factors arrive as pandas `Categorical`.
* **Downstream R scripts** (`03_create_cate_inspection_outputs.R`, figure scripts)
  read the R `.rds` objects, not these `.pkl`/`.parquet` files. To feed Python
  results into them, convert the CSV outputs.
* **Compute.** The causal-forest stage is as demanding in Python as in R
  (2000 trees on about 1 M cells, five times over). Plan to run it on a server.
  The 20–30 GB RAM figure from R is a reasonable guide.

---

## 5. Running it

```bash
pip install -r Risk/hadi_model/code_py/requirements.txt

# data/ is expected at Risk/hadi_model/data (or set LDYG_DATA_ROOT)
python Risk/hadi_model/code_py/02_models/01_fit_regression_models.py
LDYG_ALLOW_AVG_SLOPES=true python Risk/hadi_model/code_py/02_models/01_fit_regression_models.py

LDYG_ALLOW_CAUSAL_FOREST_FIT=true LDYG_ALLOW_CATE_RASTERIZE=true \
  python Risk/hadi_model/code_py/02_models/02_fit_causal_forests.py
# quick test: add LDYG_CF_NUM_TREES=200
```

Input: `data/processed/final_dataset/analysis_ready_data_regression_vars.rds`, plus,
for rasterization, `data/raw/crop_yield/yield_gap_2010_percent.tif`.

### Validation performed

The replication `data/` folder is not in this repository, so the port was
validated on a **synthetic** dataset: 3,000 cells on a 0.1° grid, every contract
column present, injected NAs, a spatially correlated error field, and known effects.

* The OLS recovered the true elasticities (0.20 / 0.10 / 0.05 / 0.00 / −0.05 →
  0.185 / 0.098 / 0.051 / −0.002 / −0.046), and every specification ran with the
  correct NA-dropped sample sizes.
* `conley_vcov` matched a brute-force dense N × N computation to 1e-15, and with
  a near-zero cutoff it reduced exactly to HC0 × (n−1)/(n−k).
* The analytic `avg_slopes` matched finite-difference derivatives of the
  `pyfixest` predictions to six decimals on the debt × debt × adaptation model.
* With 200 trees, the causal forest recovered the true heterogeneous erosion effect
  τ(x) = 1 + (clay − mean clay): correlation 0.94, mean CATE 0.96. The other four
  debts (true effect 0) came out at about 0.
* The rasterization produced six Float32 DEFLATE GeoTIFFs with no negative values,
  and the sum layer equalled the sum of the five parts.

The next step is to run both versions on the real replication data and compare
the coefficient tables and CATE maps.
