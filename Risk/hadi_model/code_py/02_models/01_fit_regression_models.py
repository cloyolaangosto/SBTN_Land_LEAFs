"""Fit the regression models used for the manuscript coefficient plots.

Python translation of ``code/02_models/01_fit_regression_models.R``.

Estimates the baseline log-log yield-gap specification and the robustness checks:
  - the baseline yield-gap model and the Fig. 2 model-specification ladder
  - nonlinear LD / adaptation average-slope objects (Fig. 2)
  - actual-yield and attainable-yield models (Fig. 3 / Supp. Fig. 6)
  - covariate and exclusion robustness checks (Supp. Fig. 3)

Each model has three SE variants, as in R:
  - main_10km/      iid SEs (fixest's default, since there are no fixed effects)
  - hc1/            heteroskedasticity-robust HC1
  - conley_100km/   Conley spatial HAC, uniform kernel, 100 km cutoff

Each model is saved as ``<name>.pkl`` (a ``FitResult``) and ``<name>_coefs.csv``.
By default, outputs go to ``data/processed/models_py/...`` so the R ``.rds``
objects are left alone.

Run from anywhere:  python Risk/hadi_model/code_py/02_models/01_fit_regression_models.py
"""

from __future__ import annotations

import itertools
import pickle
import re
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code_py/

from _shared import dataset_contract as dc  # noqa: E402
from _shared.econometrics import FitResult, avg_slopes, conley_vcov  # noqa: E402
from _shared.paths import allow_heavy, assert_files_exist, path_processed  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
dataset_path = dc.analysis_ready_dataset_path()
model_dir = path_processed("models_py", "main_10km")
model_dir_hc1 = path_processed("models_py", "hc1")
model_dir_conley100 = path_processed("models_py", "conley_100km")
for d in (model_dir, model_dir_hc1, model_dir_conley100):
    d.mkdir(parents=True, exist_ok=True)

CONLEY_CUTOFF_KM = 100.0

# --------------------------------------------------------------------------- #
# Variables (all on the __rescaled__log scale, except the categorical soil type)
# --------------------------------------------------------------------------- #
OUTCOME = dc.OUTCOME_LOG
TREATMENTS = dc.TREATMENTS_LOG

CONTROLS_environment_timeDelta_climateTerra = dc.CONTROLS_environment_timeDelta_climateTerra_LOG
CONTROLS_environment_climateTerra = dc.CONTROLS_environment_climateTerra_LOG
CONTROLS_environment_terrain = dc.CONTROLS_environment_terrain_LOG
CONTROLS_environment_soilType = dc.CONTROLS_environment_soilType
CONTROLS_environment_soilGrid = dc.CONTROLS_environment_soilGrid_LOG
CONTROLS_environment_climateWc = dc.CONTROLS_environment_climateWc_LOG
CONTROLS_adaptation = dc.CONTROLS_adaptation_LOG
CONTROLS_socioeconomic = dc.CONTROLS_socioeconomic_LOG

natural_controls = (
    CONTROLS_environment_climateWc
    + CONTROLS_environment_climateTerra
    + CONTROLS_environment_timeDelta_climateTerra
    + CONTROLS_environment_terrain
    + [CONTROLS_environment_soilType]
    + CONTROLS_environment_soilGrid
)


# --------------------------------------------------------------------------- #
# Formula handling: a tiny stand-in for R's formula objects + update()
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Spec:
    outcome: str
    terms: tuple[str, ...]

    def formula(self) -> str:
        return f"{self.outcome} ~ " + " + ".join(self.terms)

    def add(self, *terms: str) -> "Spec":           # update(f, . ~ . + x)
        return replace(self, terms=self.terms + tuple(t for t in terms if t not in self.terms))

    def drop(self, *terms: str) -> "Spec":          # update(f, . ~ . - x)
        return replace(self, terms=tuple(t for t in self.terms if t not in terms))

    def with_outcome(self, outcome: str) -> "Spec":  # update(f, y2 ~ .)
        return replace(self, outcome=outcome)

    def variables(self, columns) -> list[str]:
        """Data columns referenced by the spec (used for listwise NA deletion)."""
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", " ".join((self.outcome,) + self.terms)))
        return [c for c in columns if c in tokens]


def sq(var: str) -> str:
    return f"I({var} ** 2)"


f = Spec(OUTCOME, tuple(TREATMENTS + natural_controls + CONTROLS_adaptation + CONTROLS_socioeconomic))


def make_formula(outcome: str = OUTCOME, controls: list[str] | None = None) -> Spec:
    """Treatments plus an arbitrary control set (R: make_formula)."""
    return Spec(outcome, tuple(TREATMENTS + list(controls or [])))


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #
def estimation_sample(spec: Spec, df: pd.DataFrame, need_coords: bool) -> pd.DataFrame:
    cols = spec.variables(df.columns)
    if need_coords:
        cols = list(dict.fromkeys(cols + ["lat", "long"]))
    return df.dropna(subset=cols).reset_index(drop=True)


def fit_model(name: str, spec: Spec, df: pd.DataFrame, vcov: str = "iid") -> tuple[FitResult, pd.DataFrame]:
    """OLS with ``vcov`` in {"iid", "HC1", "conley"}.

    Returns the fit and its estimation sample (the sample is needed for avg_slopes).
    """
    data = estimation_sample(spec, df, need_coords=(vcov == "conley"))
    fit = pf.feols(spec.formula(), data=data, vcov="iid" if vcov == "conley" else vcov)
    coef = fit.coef()
    names = list(coef.index)

    if vcov == "conley":
        X = fit.model_matrix.independent[names].to_numpy(float)
        V = conley_vcov(X, np.asarray(fit.resid()), data["lat"].to_numpy(), data["long"].to_numpy(),
                        cutoff_km=CONLEY_CUTOFF_KM)
        vtype = f"conley_{int(CONLEY_CUTOFF_KM)}km"
    else:
        V = np.asarray(fit.variance_covariance.vcov)
        vtype = vcov

    res = FitResult(
        name=name, formula=spec.formula(), vcov_type=vtype, coefnames=names,
        coef=coef.to_numpy(float), vcov=V, nobs=int(fit._N), r2=float(fit._r2),
    )
    return res, data


def fit_and_save(name: str, spec: Spec, df: pd.DataFrame, vcov: str = "iid",
                 out_dir: Path = model_dir) -> FitResult:
    """Fit the model and save ``<stem>.pkl`` + ``<stem>_coefs.csv`` (R: fit_and_save)."""
    res, _ = fit_model(name, spec, df, vcov)
    save_result(res, out_dir, name)
    return res


def save_result(res: FitResult, out_dir: Path, name: str) -> None:
    stem = Path(name).stem
    with open(out_dir / f"{stem}.pkl", "wb") as fh:
        pickle.dump(res, fh)
    res.tidy().to_csv(out_dir / f"{stem}_coefs.csv", index=False)
    print(f"  saved {out_dir.name}/{stem}  (n={res.nobs}, vcov={res.vcov_type})")


def fit_avg_slopes_and_save(name: str, spec: Spec, df: pd.DataFrame, vcov: str, out_dir: Path) -> None:
    """Fit a nonlinear model, then save it and its average slopes (marginaleffects::avg_slopes)."""
    res, data = fit_model(name, spec, df, vcov)
    save_result(res, out_dir, name)
    ame = avg_slopes(res, data, TREATMENTS)
    stem = Path(name).stem + "_average_slope"
    ame.to_csv(out_dir / f"{stem}.csv", index=False)
    with open(out_dir / f"{stem}.pkl", "wb") as fh:
        pickle.dump(ame, fh)


def winsorize(x: pd.Series, lower_quantile: float = 0.025, upper_quantile: float = 0.975) -> pd.Series:
    """Cap a numeric series at its quantiles. Non-numeric series are returned unchanged.

    pandas' default 'linear' interpolation equals R's quantile(type = 7).
    """
    if not pd.api.types.is_numeric_dtype(x) or isinstance(x.dtype, pd.CategoricalDtype):
        return x
    lo, hi = x.quantile([lower_quantile, upper_quantile])
    return x.clip(lower=lo, upper=hi)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    assert_files_exist(dataset_path)
    df_ok = dc.read_rds(dataset_path)

    # Factors: crop-type dummies (R: across(crop_cols, as.factor)) and soil type / field size
    crop_cols = [c for c in df_ok.columns if c.startswith("dominantCropGerberRay_")]
    for c in crop_cols + [CONTROLS_environment_soilType, dc.ROBUSTNESS_EXTRA_CATEGORICAL]:
        if c in df_ok.columns:
            df_ok[c] = df_ok[c].astype("category")

    spec_vars = {
        "OUTCOME": OUTCOME,
        "TREATMENTS": TREATMENTS,
        "CONTROLS_environment_timeDelta_climateTerra": CONTROLS_environment_timeDelta_climateTerra,
        "CONTROLS_environment_climateTerra": CONTROLS_environment_climateTerra,
        "CONTROLS_environment_terrain": CONTROLS_environment_terrain,
        "CONTROLS_environment_soilType": CONTROLS_environment_soilType,
        "CONTROLS_environment_soilGrid": CONTROLS_environment_soilGrid,
        "CONTROLS_environment_climateWc": CONTROLS_environment_climateWc,
        "CONTROLS_adaptation": CONTROLS_adaptation,
        "CONTROLS_socioeconomic": CONTROLS_socioeconomic,
    }
    for d in (model_dir, model_dir_conley100):
        with open(d / "model_spec_variables.pkl", "wb") as fh:
            pickle.dump(spec_vars, fh)

    C = "conley"

    # ---- Baseline -----------------------------------------------------------
    fit_and_save("model_baseline", f, df_ok, "iid")
    fit_and_save("model_baseline", f, df_ok, "HC1", out_dir=model_dir_hc1)
    fit_and_save("model_baseline", f, df_ok, C, out_dir=model_dir_conley100)

    # ---- Winsorized robustness ---------------------------------------------
    df_ok_winsor = pd.DataFrame({c: winsorize(df_ok[c]) for c in df_ok.columns})
    fit_and_save("model_winsorized", f, df_ok_winsor, "iid")
    fit_and_save("model_winsorized", f, df_ok_winsor, C, out_dir=model_dir_conley100)

    # ---- Fig. 2 control ladder ---------------------------------------------
    ladder = {
        "model_no_controls": make_formula(controls=[]),
        "model_natural_controls": make_formula(controls=natural_controls),
        "model_natural_adaptation_controls": make_formula(controls=natural_controls + CONTROLS_adaptation),
        "model_full_controls": make_formula(
            controls=natural_controls + CONTROLS_adaptation + CONTROLS_socioeconomic),
    }
    for name, spec in ladder.items():
        fit_and_save(name, spec, df_ok, "iid")
    for name, spec in ladder.items():
        fit_and_save(name, spec, df_ok, C, out_dir=model_dir_conley100)

    # ---- Spatial trend: quadratic polynomial in long/lat --------------------
    f_addLongLatPoly = f.add("long", "lat", sq("long"), sq("lat"), "long:lat")
    fit_and_save("model_long_lat_polynomial", f_addLongLatPoly, df_ok, "iid")
    fit_and_save("model_long_lat_polynomial", f_addLongLatPoly, df_ok, C, out_dir=model_dir_conley100)

    # ---- Drop SoilGrids texture controls -----------------------------------
    f_rmSoilGrid = f.drop(*CONTROLS_environment_soilGrid)
    fit_and_save("model_without_soilgrid_controls", f_rmSoilGrid, df_ok, "iid")
    fit_and_save("model_without_soilgrid_controls", f_rmSoilGrid, df_ok, C, out_dir=model_dir_conley100)

    # ---- Crop-type dummies (saved under two names, as in R) -----------------
    f_addCropType = f.add(*dc.crop_type_dummy_vars)
    for d, v in ((model_dir, "iid"), (model_dir_conley100, C)):
        res = fit_and_save("model_yield_gap_crop_type_controls", f_addCropType, df_ok, v, out_dir=d)
        save_result(replace(res, name="model_crop_type_controls"), d, "model_crop_type_controls")

    # ---- Supp. Fig. 3: extra covariates / exclusions -----------------------
    L = dc.LOG_SUFFIX
    extra = {
        "model_field_size_control": f.add(dc.ROBUSTNESS_EXTRA_CATEGORICAL),
        "model_soil_depth_control": f.add("soilDepth__mean" + L),
        "model_soil_n_control": f.add("nitrogen__sum" + L),
        "model_water_table_depth_control": f.add("waterTableDepth" + L),
        "model_road_density_control": f.add("roadDens__mean" + L),
        "model_travel_time_to_cities_control": f.add("travel_time_to_cities" + L),
        "model_without_machinery_control": f.drop("machinery_ca2010" + L),
        "model_agricultural_expenditure_control": f.add(
            "agriShareOfGovtExpend_ca2010" + L, "agriOrientIdxForGovtExpend_ca2010" + L),
        "model_economic_freedom_control": f.add("efw_ca2010" + L),
    }
    for name, spec in extra.items():
        fit_and_save(name, spec, df_ok, "iid")

    # ---- Fig. 3 / Supp. Fig. 6: actual & attainable yield outcomes ----------
    actual, attain = "actualYieldGerberRay" + L, "attainableYieldGerberRay" + L
    fit_and_save("model_actual_yield", f.with_outcome(actual), df_ok, "iid")
    fit_and_save("model_attainable_yield", f.with_outcome(attain), df_ok, "iid")
    fit_and_save("model_actual_yield_crop_type_controls", f_addCropType.with_outcome(actual), df_ok, "iid")
    fit_and_save("model_attainable_yield_crop_type_controls", f_addCropType.with_outcome(attain), df_ok, "iid")

    # ---- Nonlinear models + average slopes (heavy, opt-in) ------------------
    if allow_heavy("LDYG_ALLOW_AVG_SLOPES"):
        f_addQuadraticLDs = f.add(*[sq(t) for t in TREATMENTS])

        # all 2-, 3-, 4- and 5-way products of the five debts (R: combn)
        interactions_LDs = [
            ":".join(c) for i in range(2, len(TREATMENTS) + 1)
            for c in itertools.combinations(TREATMENTS, i)
        ]
        f_addInteractionLDs = f.add(*interactions_LDs)

        interactions_LDsAdapt = [f"{t}:{a}" for t in TREATMENTS for a in CONTROLS_adaptation]
        f_addInteractionLDsAdapt = f.add(*interactions_LDs, *interactions_LDsAdapt)

        nonlinear = {
            "model_quadratic_land_degradation": f_addQuadraticLDs,
            "model_interaction_land_degradation": f_addInteractionLDs,
            "model_interaction_land_degradation_adaptation": f_addInteractionLDsAdapt,
        }
        for name, spec in nonlinear.items():
            fit_avg_slopes_and_save(name, spec, df_ok, "iid", model_dir)
            fit_avg_slopes_and_save(name, spec, df_ok, C, model_dir_conley100)
    else:
        print("Skipped average slopes. Set LDYG_ALLOW_AVG_SLOPES=true to recompute "
              "the nonlinear-model average-slope objects.")


if __name__ == "__main__":
    main()
