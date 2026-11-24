# From Hadi et al. to Thresholds × LEAFs: a study guide and design note

**Goal.** Re-use the Hadi et al. design (*"Land degradation is associated with larger crop yield gaps
across global croplands"*) to estimate how far a place sits from its SBTN **threshold** (as seen through
the **LEAFs**) is associated with crop yield gaps for a few selected commodities. Hadi's five
"degradation debts" are replaced by threshold–LEAF distances.

**Assumed background.** Master's-level statistics, rusty after about ten years. You already know what a
regression, a variance and a p-value are. This guide refreshes the parts that matter here and skips the rest.

**How the guide is organised.**

1. [Concepts, each with a tiny simulated example](#part-1--concepts-with-small-examples): OLS → heteroskedasticity (HC1) → spatial correlation (Conley) → causal inference → random forests → causal forests
2. [How the Hadi model uses each piece](#part-2--how-the-hadi-model-uses-each-piece)
3. [Switching from debts to threshold–LEAF distances](#part-3--switching-to-thresholdleaf-distances): what changes, the traps, and a proposed specification
4. [Reading and watching list](#part-4--resources)

All examples run in the project `.venv` (pyfixest 0.70, econml 0.17, scikit-learn 1.7). The printed
outputs come from actual runs.

---

## Part 1 — Concepts with small examples

### 1.1 OLS, the 5-minute refresher

$$y_i = x_i'\beta + \varepsilon_i,\qquad \hat\beta = (X'X)^{-1}X'y$$

* $\hat\beta_k$ is the change in $E[y]$ for a one-unit change in $x_k$, **holding the other regressors fixed**.
* **Log–log model.** If both $y$ and $x_k$ are logged, $\beta_k$ is an **elasticity**: a 1 % increase in
  $x_k$ goes with a $\beta_k$ % change in $y$. Hadi's regressions are log–log.
* The **classical ("iid") variance** $\hat V = \hat\sigma^2 (X'X)^{-1}$ assumes the errors are independent
  *and* have the same variance. Both assumptions fail for global grid data. Sections 1.2 and 1.3 are about that.

```python
import numpy as np, pandas as pd, pyfixest as pf
rng = np.random.default_rng(0)
n = 500
x = rng.normal(size=n)
y = 2 + 0.5 * x + rng.normal(size=n)

X = np.column_stack([np.ones(n), x])
beta = np.linalg.solve(X.T @ X, X.T @ y)            # (X'X)^-1 X'y
u = y - X @ beta
se_iid = np.sqrt(np.diag(u @ u / (n - 2) * np.linalg.inv(X.T @ X)))

fit = pf.feols("y ~ x", data=pd.DataFrame({"y": y, "x": x}))
```
```
numpy     beta [1.931 0.505]  se [0.042 0.041]
pyfixest  beta [1.931 0.505]  se [0.042 0.041]
```

pyfixest is doing nothing magical. It is the same algebra, plus formula parsing, factor handling and
variance options.

### 1.2 Heteroskedasticity and HC1 ("robust") standard errors

**Problem.** If the error variance changes with $x$ (for example, yield gaps are much noisier in arid
regions), $\hat\beta$ is still unbiased, but $\hat\sigma^2(X'X)^{-1}$ gives the **wrong standard error**.

**Fix: the sandwich.**

$$\hat V_{HC} = (X'X)^{-1}\Big[\textstyle\sum_i \hat u_i^2\, x_i x_i'\Big](X'X)^{-1}$$

The "bread" is $(X'X)^{-1}$. The "meat" lets every observation have its own error variance $\hat u_i^2$.
**HC0** is exactly this. **HC1** multiplies it by $n/(n-k)$, a degrees-of-freedom correction like
dividing by $n-k$ instead of $n$. With $n \approx 400{,}000$ the HC0/HC1 difference is negligible. What
matters is robust versus classical.

```python
b_hat, se_i, se_h = [], [], []
for r in range(2000):                                   # Monte Carlo: repeat the "study" 2000 times
    x = rng.uniform(0, 3, n)
    y = 1 + 0.5 * x + rng.normal(size=n) * x            # noise grows with x
    X = np.column_stack([np.ones(n), x])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y; u = y - X @ b
    V_iid = (u @ u / (n - 2)) * XtXi
    V_hc1 = n / (n - 2) * XtXi @ ((X * u[:, None]**2).T @ X) @ XtXi
    b_hat.append(b[1]); se_i.append(np.sqrt(V_iid[1, 1])); se_h.append(np.sqrt(V_hc1[1, 1]))
```
```
true sd of slope: 0.1004 | mean iid SE: 0.0893 | mean HC1 SE: 0.0977
```

"True sd" is how much $\hat\beta$ really varies across repeated samples. HC1 tracks it, while iid is too
small. In pyfixest this is `pf.feols(..., vcov="HC1")`.

### 1.3 Spatial correlation and Conley standard errors

**Problem.** Neighbouring 10 km cells share weather, soils and farming practices, so their errors are
**correlated**. HC1 still assumes independence. With positively correlated errors and a spatially smooth
regressor, you effectively have far fewer independent observations than $n$, and iid or HC1 SEs can be
**several times too small**.

**Step 1: clustering.** If you can draw boxes (clusters $g$) so that errors correlate inside a box but
not across boxes, sum the scores within each box before squaring:

$$\text{meat} = \textstyle\sum_g \big(\sum_{i\in g} x_i\hat u_i\big)\big(\sum_{i\in g} x_i\hat u_i\big)'$$

Typical boxes are countries or ecoregions. The weakness is borders: two cells 5 km apart in different
boxes are treated as independent.

**Step 2: Conley (1999).** Drop the boxes. Every pair of cells closer than a cutoff distance gets weight:

$$\text{meat} = \textstyle\sum_i\sum_j K(d_{ij})\,(x_i\hat u_i)(x_j\hat u_j)',\qquad K(d)=\mathbb 1[d\le c]\ \text{(uniform, as in Hadi)}\ \text{or}\ \max(0,1-d/c)\ \text{(Bartlett)}$$

With $c = 0$ this collapses back to HC0. The cutoff $c$ should be at least as large as the distance over
which residuals are still correlated. Hadi uses 100 km.

```python
from scipy.ndimage import gaussian_filter
def smooth_field(side, sigma):                     # spatially autocorrelated noise on a grid
    f = gaussian_filter(rng.normal(size=(side, side)), sigma)
    return (f / f.std()).ravel()

side = 40
gx, gy = np.meshgrid(np.arange(side), np.arange(side))
coords = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
D  = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))   # pairwise distances (cells)
Ku = (D <= 6).astype(float)                                     # uniform kernel, cutoff 6 cells

for r in range(300):
    x = smooth_field(side, 1.5); e = smooth_field(side, 1.5)    # both x and error are smooth
    y = 1 + 0.5 * x + e
    X = np.column_stack([np.ones(x.size), x])
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; u = y - X @ b
    s = X * u[:, None]                                          # scores x_i * u_i
    V_conley = XtXi @ (s.T @ Ku @ s) @ XtXi
```
```
true sd 0.0865 | iid 0.0249 | HC1 0.0256 | Conley uniform 0.0838 | Conley Bartlett 0.0725
```

iid and HC1 are **3.5× too small**. Conley gets close to the truth. This is why Hadi reports Conley
SEs, and it will matter even more for you (see §3.3e).

Two practical notes:

* With a uniform kernel the Conley "variance" can occasionally come out **negative** (it isn't
  guaranteed positive semi-definite). This happened in an earlier, more extreme version of this
  simulation. Bartlett avoids most of it, at the cost of being a bit smaller.
* Conley needs pairwise distances. On about 400k cells, the Hadi port
  ([`_shared/econometrics.py`](hadi_model/code_py/_shared/econometrics.py), `conley_vcov`) uses a
  haversine BallTree and sparse chunks instead of the dense matrix above.

### 1.4 Causal inference: when is a coefficient an *effect*?

**Potential outcomes.** For cell $i$, $Y_i(w)$ is the yield gap it *would* have at debt level $w$. A causal
effect compares $Y_i(w')$ with $Y_i(w)$. We only ever see one of them, so we need an assumption that
lets observed differences between cells stand in for the unseen comparison.

**Selection on observables (unconfoundedness).** Conditional on controls $Z$, the treatment is as good as
randomly assigned. Every observational analysis in the Hadi paper, OLS and causal forest alike, relies on
this. That is why the title says *"associated with"*.

Three things to keep in mind, each with a simulation:

**(a) Confounders must be controlled.** Aridity drives both erosion and yield gaps:

```python
n = 5000
aridity = rng.normal(size=n)
erosion = 0.8 * aridity + rng.normal(size=n)
yg      = 0.3 * erosion + 1.0 * aridity + rng.normal(size=n)       # true effect 0.3
pf.feols("yg ~ erosion", df); pf.feols("yg ~ erosion + aridity", df)
```
```
naive: 0.81 | controlled: 0.316   (true 0.3)
```

**(b) Mediators ("bad controls") must *not* be controlled if you want the total effect.** Suppose
erosion makes farmers apply less fertilizer, which raises the yield gap further:

```python
fert = -0.5 * erosion + rng.normal(size=n)
yg2  = 0.3 * erosion - 0.4 * fert + 1.0 * aridity + rng.normal(size=n)   # total = 0.3 + 0.2 = 0.5
```
```
total effect: 0.505 | with mediator fert in the model: 0.303
```

Neither number is wrong. They answer different questions (total effect versus direct effect).
Hadi's **control ladder** (no → natural → +adaptation → +socio-economic controls) is there to show how
much the answer depends on this choice.

**(c) Overlap (positivity).** You can only compare cells with similar $Z$ if, at that $Z$, there is still
variation in the treatment. If the treatment is (nearly) a function of the controls, nothing is left to
learn from. Remember this one. It is the biggest issue for LEAFs (§3.3c).

Drawing a **DAG** (a box-and-arrow diagram of what causes what) before choosing controls is the
cheapest insurance against (a) and (b). [DAGitty](https://www.dagitty.net) is a free browser tool for it.

### 1.5 Random forests

A **regression tree** splits the data into boxes ("if clay > 30 % and aridity < 0.4 …") and predicts the
mean of $y$ in each box. A single tree overfits. A **random forest** averages many trees, each grown on a
random subsample of rows and choosing splits from a random subset of columns
(`max_features`, called `mtry` in R). Averaging de-correlated trees cuts the variance.

Things you need for the Hadi pipeline:

* **Out-of-bag (OOB) predictions.** Each tree never sees part of the data. Predicting each row only from
  the trees that didn't train on it gives an honest out-of-sample prediction without a separate test set.
  Hadi's nuisance estimates $\hat m(X)$ and $\hat e(X)$ are OOB.
* **Hyperparameters**: `n_estimators` (number of trees), `min_samples_leaf` (smallest box, 5 in Hadi),
  `max_features`, and `max_samples` (subsample fraction, 0.5 in grf).
* **Strengths.** Nonlinearities and interactions come for free, and monotone transforms of $X$ don't matter.
  **Weaknesses.** It can't extrapolate beyond the data, and its importances aren't effects.

```python
from sklearn.ensemble import RandomForestRegressor
Xr = rng.uniform(0, 1, size=(3000, 5))
yr = np.where(Xr[:, 0] > 0.5, 2.0, 0.0) + np.sin(6 * Xr[:, 1]) + rng.normal(scale=0.5, size=3000)
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.6,
                           oob_score=True, random_state=0).fit(Xr, yr)
```
```
OOB R2: 0.853 | importances: [0.63 0.32 0.02 0.02 0.01]
linear regression CV R2: 0.625
```

The forest finds the step in $x_0$ and the wave in $x_1$ and ignores the three noise columns.

> **Spatial caveat.** With spatial data, random or OOB validation is **over-optimistic**, because a test
> cell's neighbours are in the training set. Use spatial block cross-validation when you report predictive
> skill (Ploton et al. 2020; Roberts et al. 2017).

### 1.6 Causal forests

A random forest predicts $Y$. A **causal forest** estimates how the *effect* of $W$ on $Y$ varies across
places, $\tau(x)$ (the **CATE**, conditional average treatment effect). It does this in three steps:

1. **Residualise (double ML / R-learner).** Fit two ordinary forests, $\hat m(X)\approx E[Y\mid X]$ and
   $\hat e(X)\approx E[W\mid X]$. Work with $\tilde Y = Y-\hat m$ and $\tilde W = W-\hat e$. This removes the
   part of $Y$ and $W$ that the confounders explain, like "controlling for $X$" but flexibly. It is also
   robust to small errors in the two forests (*Neyman orthogonality*).
2. **Split for heterogeneity.** Grow trees whose splits maximise differences in the slope of
   $\tilde Y$ on $\tilde W$, not prediction accuracy.
3. **Honesty.** Each tree chooses its splits on one half of its sample and estimates the leaf effects on
   the other half. This is what makes the confidence intervals valid.

For a point $x$, $\hat\tau(x)$ is a **weighted local regression** of $\tilde Y$ on $\tilde W$, where the
weights are how often each training cell shares a leaf with $x$.

```python
from econml.dml import CausalForestDML
n = 4000
clay, aridity = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
Xc  = np.column_stack([clay, aridity, rng.uniform(0, 1, (n, 3))])
W   = 0.8 * aridity + rng.normal(scale=0.5, size=n)        # debt, confounded by aridity
tau = 0.2 + 0.6 * clay                                      # effect is larger on clay soils
Y   = tau * W + 1.0 * aridity + rng.normal(scale=0.5, size=n)

cf = CausalForestDML(model_y=RandomForestRegressor(min_samples_leaf=20),
                     model_t=RandomForestRegressor(min_samples_leaf=20),
                     n_estimators=500, min_samples_leaf=10, random_state=0).fit(Y, W, X=Xc)
tau_hat = cf.effect(Xc); lo, hi = cf.effect_interval(Xc)
```
```
corr(tau_hat, tau): 0.953 | mean tau_hat: 0.484 (true 0.502) | naive OLS slope: 0.697
first 5 95% CIs: [0.17,0.37] [0.24,0.45] [0.52,0.93] [0.50,0.75] [0.45,0.77]
true:             0.27        0.41        0.78        0.70        0.69
```

The forest recovers both the confounding-corrected average and the "larger on clay" pattern.

`CausalForestDML` bundles the three steps. Hadi's port does them by hand with `econml.grf.RandomForest`
and `CausalForest` to mirror R's `grf` settings exactly. The idea is the same.

> **Spatial caveat (again).** The forest's CIs assume independent cells, so with spatially correlated
> data they are too narrow, for the same reason as §1.3. Read CATE maps for *pattern*, not
> cell-by-cell significance.

---

## Part 2 — How the Hadi model uses each piece

| Concept | Where in Hadi | Details |
|---|---|---|
| Log–log OLS | `01_fit_regression_models` | $\log YG = \alpha + \sum_k \beta_k\log D_k + \gamma'Z + \varepsilon$, all five debts together. Each $\beta_k$ is an elasticity net of the other debts |
| Variables rescaled to [1, 101] before logging | data pipeline | Keeps $\log$ defined where a debt is exactly 0 (common after clipping negatives) |
| iid / HC1 / Conley 100 km | three model folders | Same coefficients, three uncertainty estimates. Conley is the one to trust |
| Control ladder, bad controls | `model_no_controls` … `model_full_controls` | Adaptation and socio-economic controls may be mediators (§1.4b) |
| Robustness | winsorised, lat/long polynomial, extra covariates, crop dummies | Outliers, smooth spatial trends, omitted variables |
| Nonlinearity | quadratic and interaction models + `avg_slopes` | Average marginal effect, comparable to the linear $\beta$ |
| Outcome decomposition | actual vs attainable yield | Does degradation lower actual yield or raise the benchmark? |
| Causal forest | `02_fit_causal_forests` | One forest per debt. $W$ = that debt, $X$ = controls + other four debts, 2000 trees, `min_samples_leaf=5` |
| CATE maps | Stage B | OOB CATEs rasterised; negatives set to 0; the five layers summed |
| Loss accounting | `04_compute_loss_rasters_and_tables` | CATE × debt → production / calorie / revenue losses |

Read [code_py/README.md](hadi_model/code_py/README.md) and the walkthrough notebook
[hadi_model_walkthrough.ipynb](hadi_model/code_py/hadi_model_walkthrough.ipynb) alongside this table.

---

## Part 3 — Switching to threshold–LEAF distances

### 3.1 What stays, what changes

| Element | Hadi | Your version |
|---|---|---|
| Unit | ~10 km cropland cell | ~10 km cell **where the selected commodity is grown** |
| Outcome | aggregate yield gap 2010 (%) | **commodity-specific** yield gap (e.g. GAEZ v4 actual vs attainable yield, MapSPAM harvested area and yield) |
| Treatments | 5 debts (change vs a past state) | signed **distance to threshold** per indicator: SOC, soil erosion, acidification |
| Controls | climate, terrain, soil, adaptation, socio-economic | same groups, but **check each against the LEAF's own model inputs** (§3.3c) |
| Methods | OLS + iid/HC1/Conley, causal forest | the same, plus a **threshold (hinge) term** and ecoregion-aware SEs |
| Code | Hadi port | reuse `Spec`, `fit_model`, `conley_vcov`, `avg_slopes` and the causal-forest functions directly |

### 3.2 Defining the treatment

Use one sign convention: **positive = on the wrong side of the threshold**.

| Indicator | Threshold in the repo | LEAF | Signed distance $g$ |
|---|---|---|---|
| SOC | [`thresholds/SOC_TH.csv`](../thresholds/SOC_TH.csv), t C/ha per ecoregion | SOC stock t C/ha (2030, per land-use flow) | $g = T_e - \text{SOC}$ (below threshold is bad) |
| Soil erosion | **not in the repo yet**; needs a tolerable soil-loss rate, e.g. Verheijen et al. 2009 | t soil/ha/yr per flow | $g = \text{SE} - T$ (above tolerable is bad) |
| Acidification | [`thresholds/acid_thresholds.csv`](../thresholds/acid_thresholds.csv): N-deposition threshold vs ecoregion baseline | a **characterisation factor** (kg SO₂-eq per kg emitted), not a state | $g = \text{deposition} - T_e$ (the csv's "Threshold exceedance" column) |

Notes:

* The threshold csvs already store a **"Threshold exceedance"** column with this sign convention
  (SOC: threshold − baseline; acidification: baseline − threshold). That makes a natural ecoregion-level
  treatment.
* **Acidification is different in kind.** The LEAF says how much acidification one more kg of emission
  causes. It doesn't describe the soil's current state. The state-like quantity is deposition versus its
  critical threshold. Use the deposition exceedance as the treatment, and keep the LEAF for translating a
  company's emissions into a change in $g$ afterwards.
* Clipping $g$ at 0, as Hadi does with debts, throws away the "how safe" information on the good side.
  Keep the signed $g$ and add the hinge term in §3.4.
* **Don't log a signed distance.** Hadi's rescale-then-log works for non-negative debts. For $g$, use
  levels (standardised, or in natural units such as t C/ha), or `asinh` if the tails are heavy. The
  coefficient is then a semi-elasticity or a level effect, not an elasticity.

### 3.3 Design traps, and how to handle each

**(a) Timing: the LEAFs describe 2030; the outcome is about 2010–2020.**
SOC LEAFs are RothC projections to 2030 under constant land use. A 2030 value can't cause a 2010 yield
gap. Using it as a regressor mixes a forecast into the treatment.

*Recommendation: estimate with observed states, project with LEAFs.*

* **Estimation:** $g$ built from an observed baseline that predates or matches the outcome year (the
  SoilGrids SOC baseline that seeds the LEAFs, observed erosion, or 2011–2020 deposition), measured
  against the threshold.
* **Projection:** apply the fitted effects (the OLS $\beta$ or the CATE map) to the **LEAF-implied**
  distance $g^{\text{LEAF}}$ for each commodity and practice, to get the expected yield-gap change "if
  this land-use / management continues to 2030". This mirrors Hadi's step `04` (CATE × debt → loss), with
  LEAFs supplying the future debt.

**(b) Thresholds are constant within an ecoregion.**
If $g = \text{SOC} - T_e$ (or its reverse) and you include ecoregion fixed effects, $T_e$ is absorbed
completely. The coefficient on $g$ is then *identical* to the coefficient on SOC alone:

```python
n_eco, per = 60, 100
eco = np.repeat(np.arange(n_eco), per)
T   = rng.uniform(40, 80, n_eco)[eco]               # SOC threshold, one per ecoregion
soc = T + rng.normal(0, 15, eco.size)
g   = T - soc                                       # positive = below threshold (bad)
exceed = np.clip(g, 0, None)
yg  = 30 + 0.4 * exceed + rng.normal(0, 5, eco.size)   # harm only once below threshold
pf.feols("yg ~ soc | eco", df); pf.feols("yg ~ g | eco", df)
pf.feols("yg ~ g + exceed | eco", df, vcov={"CRV1": "eco"})
```
```
yg ~ soc | eco          soc     = -0.200
yg ~ g   | eco          g       = +0.200          (same information, sign flipped)
yg ~ g + exceed | eco   g       = -0.007 (se 0.007)
                        exceed  = +0.415 (se 0.013)   (true: 0 above threshold, 0.4 per t below)
```

So **the threshold only adds information through nonlinearity**: does the yield gap respond differently
once you cross it? That is the hypothesis behind using thresholds at all, and it's directly testable.
A linear model on $g$ reports a blurred "0.2", which is right on neither side.

**(c) LEAFs are built from your control variables.**
Soil-erosion LEAFs come from RUSLE, $A = R\cdot K\cdot LS\cdot C\cdot P$: rainfall erosivity (from
precipitation), soil erodibility (from texture, which in this repo comes from HWSD), slope, and land-use
and management factors. SOC LEAFs come from RothC, driven by climate, clay and inputs. If you control for
precipitation, texture and slope as Hadi does, **little independent variation in the LEAF is left**. That
is the overlap problem from §1.4c: estimates become unstable or driven by functional form.

What to do:

* **Write the LEAF's own formula into your DAG.** Inputs of the LEAF model are *causes of the treatment*.
  Some are also confounders (precipitation affects yield directly), and those must stay. Others affect
  yield only *through* the soil state, and those can be dropped as controls.
* **Diagnose overlap.** Regress $g$ on the controls. An $R^2$ near 1 is a red flag. In the causal forest,
  inspect $\hat e(X)$: a residual $\tilde W$ with almost no variance means no identifying information.
* **Prefer observed states** (point a). Observed SOC or erosion carries real variation beyond what the
  inputs predict. A modelled LEAF largely doesn't.

**(d) Commodity and practice matching.**
LEAFs vary by flow (crop × irrigation × tillage × residues). For each cell, choose the flow that matches
what is actually there, using MapSPAM irrigated/rainfed shares, and keep the sample to cells where the
commodity has meaningful harvested area. Run **one model per commodity** rather than pooling. If you
pool, add commodity fixed effects and commodity × $g$ interactions.

**(e) Resolution and standard errors.**

* SOC and acidification rasters are about 9–10 km, soil erosion 25 km, thresholds per ecoregion. After
  resampling to 10 km, each erosion value repeats about 6 times and each threshold thousands of times.
  The effective sample size is much smaller than the number of cells.
* Report **Conley SEs with a cutoff well above 25 km** (keep Hadi's 100 km and try 200–300 km as a
  check), **plus ecoregion-clustered SEs** (`vcov={"CRV1": "ECO_ID"}`), because the threshold varies only
  at that level. Trust the larger of the two.
* For causal-forest CIs, use spatial blocks: pass `groups=` (for example 1° tiles or ecoregions) to
  `CausalForestDML.fit` so neighbouring cells are kept in the same cross-fitting fold, and treat the
  pointwise CIs as lower bounds on the true uncertainty.

**(f) Bad controls, revisited.** Fertilizer, irrigation and tillage affect both SOC/erosion and yields.
Keep Hadi's ladder and report every rung. For SOC especially, management is often the *mechanism*
through which the threshold gets crossed.

### 3.4 Proposed specification

For commodity $c$, cell $i$, ecoregion $e(i)$, indicators $k \in \{\text{SOC}, \text{SE}, \text{Acid}\}$:

**OLS with a threshold kink (the core model):**

$$YG_{ic} = \alpha + \sum_k\big[\beta_k\, g_{ik} + \delta_k\, \max(g_{ik},0)\big] + \gamma' Z_i + \varepsilon_{ic}$$

* $\beta_k$ is the slope on the safe side; $\beta_k+\delta_k$ is the slope beyond the threshold.
* $H_0: \delta_k = 0$ is the test of **"does the threshold matter?"**
* Variants: add ecoregion fixed effects (absorbs anything constant at the ecoregion level, including
  $T_e$; the kink is still identified); add a 0/1 term $\mathbb 1[g_{ik}>0]$ for a jump at the threshold;
  plot a binned scatter of $\tilde{YG}$ against $g$ (both residualised on $Z$) to *see* the kink before
  you trust it.
* Run it with iid, HC1, Conley 100 km and ecoregion-clustered SEs, across the control ladder, as Hadi does.

**Causal forest per indicator:**

* $W = g_{ik}$ (signed), $X$ = controls + the other indicators' $g$ (as in Hadi).
* The CATE map shows **where** crossing or approaching the threshold matters most for commodity $c$.
* Plot $\hat\tau(x)$ against $g$: if the threshold matters, effects should be larger for cells near or
  beyond it.
* Use `econml.dml.CausalForestDML` for a first pass. Switch to the Hadi port's manual version if you need
  grf-equivalent settings.

**Projection to LEAF scenarios:**

$$\Delta \widehat{YG}_{ic} = \hat f\big(g^{\text{LEAF}}_{ik}\big) - \hat f\big(g^{\text{obs}}_{ik}\big)$$

Here $\hat f$ is the fitted hinge model, or the CATE times the change in $g$. Compare management
flows (conventional vs reduced tillage, residues on vs off) to show the yield-gap side of each practice.

**Minimal code sketch** (reusing the Hadi port):

```python
import pyfixest as pf
from _shared.econometrics import conley_vcov

for k in ["soc", "se", "acid"]:
    df[f"exc_{k}"] = df[f"g_{k}"].clip(lower=0)

rhs  = " + ".join([f"g_{k} + exc_{k}" for k in ["soc", "se", "acid"]] + CONTROLS)
data = df.dropna(subset=["yg_wheat", "lat", "long", "ECO_ID", *rhs.split(" + ")])  # one sample for all SEs

fit_cl = pf.feols(f"yg_wheat ~ {rhs}", data=data, vcov={"CRV1": "ECO_ID"})   # ecoregion-clustered
fit    = pf.feols(f"yg_wheat ~ {rhs}", data=data, vcov="iid")
names  = list(fit.coef().index)
X      = fit.model_matrix.independent[names].to_numpy(float)                  # as in fit_model()
V_conley = conley_vcov(X, np.asarray(fit.resid()), data["lat"].to_numpy(),
                       data["long"].to_numpy(), cutoff_km=100)
```

(`CONTROLS` must hold plain column names for the `dropna` line to work. With factor controls, use the
port's `estimation_sample()` instead.)

### 3.5 Step-by-step plan

1. **Choose 2–4 commodities** where LEAFs, thresholds and good yield-gap data overlap (e.g. wheat, maize,
   soy, palm oil). Note the outcome source and year for each.
2. **Build the grid table** on Hadi's 10 km grid: outcome, harvested area, observed indicator states,
   thresholds joined by `ECO_ID`, $g$ and $\max(g,0)$, LEAF-based $g^{\text{LEAF}}$ for each management
   flow, and Hadi's controls.
3. **Write the DAG**, including the RUSLE and RothC inputs. Decide which controls are confounders and
   which are mediators.
4. **Overlap diagnostics**: $R^2$ of $g$ on $Z$; maps of $g$; the share of cells beyond each threshold
   per commodity.
5. **OLS hinge models** with the four SE types and the control ladder. Make binned scatters of the kink.
6. **Causal forests** per commodity and indicator; CATE maps; $\hat\tau$ against $g$.
7. **Projection** with LEAF-based distances for each management flow → expected yield-gap change.
8. **Robustness**: Conley cutoff 100/200/300 km; drop ecoregions with few cells; winsorise; spatial
   block CV for the nuisance forests.

---

## Part 4 — Resources

Ordered within each block from easiest to deepest. All are free unless marked.

### Main text

* **Chernozhukov, Hansen, Kallus, Spindler & Syrgkanis, *Applied Causal Inference Powered by ML and AI*.**
  causalml-book.org, with Python notebooks. Covers OLS inference, ML, double ML and heterogeneous effects
  in one book. Syrgkanis co-created econml.

### OLS and standard errors (§1.1–1.3)

* Huntington-Klein, *The Effect*, ch. 13 (theeffectbook.net). Regression and robust/clustered SEs, intuitively.
* Long & Ervin (2000), *The American Statistician*. HC0–HC3 explained; what the "1" in HC1 is.
* Bergé, "On standard errors", `fixest` vignette. Exactly how fixest/pyfixest compute each SE type.
* Cameron & Miller (2015), "A Practitioner's Guide to Cluster-Robust Inference", *J. Human Resources*.
* Colella, Lalive, Sakalli & Thoenig (2019), "Inference with Arbitrary Clustering", IZA DP 12584. Conley as generalised clustering.
* Kelly (2019), "The Standard Errors of Persistence". Spatial noise can look significant; important for smooth global layers.
* Conley (1999), *J. Econometrics*. The original; optional.

### Causal inference (§1.4)

* Cunningham, *Causal Inference: The Mixtape* (mixtape.scunning.com). Potential outcomes, DAGs, selection on observables.
* Hernán & Robins, *Causal Inference: What If*, Part I.
* Brady Neal, *Introduction to Causal Inference*. Lecture series on YouTube with a free book; written for ML people.

### Random forests and ML (§1.5)

* StatQuest (YouTube): decision trees, random forests parts 1–2, bias–variance, cross-validation.
* James, Witten, Hastie & Tibshirani, *ISLP* (Python edition, statlearning.com), ch. 8, plus Hastie & Tibshirani's "Statistical Learning" videos.
* Hastie, Tibshirani & Friedman, *Elements of Statistical Learning*, ch. 15 (theory).
* Breiman (2001), "Random Forests" and "Statistical Modeling: The Two Cultures".

### Causal forests (§1.6)

* Athey & Wager (2019), "Estimating Treatment Effects with Causal Forests: An Application", *Observational Studies*. Short and applied; the same workflow as Hadi.
* grf documentation, "The GRF Algorithm" (grf-labs.github.io/grf), and the econml docs for `CausalForestDML` and `econml.grf`.
* Stanford GSB, *Machine Learning & Causal Inference: A Short Course* (Athey, Wager et al.), videos plus R tutorials.
* Wager & Athey (2018, *JASA*); Athey, Tibshirani & Wager (2019, *Annals of Statistics*). Theory.
* Facure, *Causal Inference for the Brave and True*. Light, Python-first.

### Spatial and GIS (§1.3, §3.3)

* Rey, Arribas-Bel & Wolf, *Geographic Data Science with Python* (geographicdata.science/book). Spatial weights, autocorrelation, spatial regression.
* Luc Anselin lectures, Center for Spatial Data Science (YouTube). Spatial autocorrelation and spatial regression.
* Reich et al. (2021), "A Review of Spatial Causal Inference Methods for Environmental and Epidemiological Applications", *Int. Statistical Review*.
* Paciorek (2010), "The Importance of Scale for Spatial-Confounding Bias…", *Statistical Science*.
* Ploton et al. (2020), *Nature Communications*; Meyer & Pebesma (2021), *Methods Ecol. Evol.*; Roberts et al. (2017), *Ecography*. Spatial validation for RF maps.
* *Geocomputation with R*, statistical-learning chapter. RF with spatial CV.

### Environmental and conservation applications

* Ferraro & Hanauer (2014), *Annual Review of Resource Economics*. Causal inference for environmental programmes.
* Schleicher et al. (2020), "Statistical Matching for Conservation Science", *Conservation Biology*.
* Verheijen et al. (2009), "Tolerable versus actual soil erosion rates in Europe", *Earth-Science Reviews*. A starting point for an erosion threshold.

### Suggested order (about 3–4 weeks part-time)

1. *The Effect* ch. 13 → Long & Ervin → Cameron & Miller → Colella et al. (re-run §1.2–1.3)
2. Mixtape: potential outcomes and matching, then draw the DAG for §3.3c
3. StatQuest RF videos → ISLP ch. 8
4. CausalML book: double ML and heterogeneous-effects chapters → Athey & Wager (2019)
5. Reich et al. (2021) + Kelly (2019) before interpreting anything spatial
