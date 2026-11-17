"""Fit the causal forests that estimate heterogeneous yield-gap responses.

Python translation of ``code/02_models/02_fit_causal_forests.R``.

It fits one causal forest per land-degradation debt, each with a continuous
treatment. Each forest gives a conditional average treatment effect (CATE) per
grid cell: the local slope d(yield gap %)/d(debt), conditional on all controls
and on the other four debts. The CATE tables are then rasterized onto the
yield-gap grid, negative values are clipped to 0, and the five layers are summed.

R ``grf::causal_forest`` is reproduced with the econml GRF building blocks:
  1. local centering: honest ``econml.grf.RegressionForest`` fits for
     E[Y|X] and E[W|X], with out-of-bag predictions (as in grf's Y.hat / W.hat)
  2. ``econml.grf.CausalForest`` (criterion="het", honest, fit_intercept=True)
     on the centered Y and W
  3. out-of-bag CATE predictions (R: ``predict(cf)`` with no newdata)

Both stages are opt-in, as in R:
  LDYG_ALLOW_CAUSAL_FOREST_FIT=true   fit the forests and write the CATE tables
  LDYG_ALLOW_CATE_RASTERIZE=true      rasterize the saved CATE tables

Python-only overrides, intended for quick tests:
  LDYG_CF_NUM_TREES (default 2000), LDYG_CF_NUM_THREADS (default 4)

Outputs go to ``data/processed/causal_forest_py/``.
"""

from __future__ import annotations

import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code_py/

from _shared import dataset_contract as dc  # noqa: E402
from _shared.paths import (  # noqa: E402
    allow_heavy, assert_files_exist, path_processed, path_raw, stop_unless_heavy_allowed,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
dataset_path = dc.analysis_ready_dataset_path()
yield_gap_template_path = path_raw("crop_yield", "yield_gap_2010_percent.tif")
cf_dir = path_processed("causal_forest_py")
cf_raster_dir = cf_dir / "rasterized"

# short label -> (treatment column, CATE column name used in the merged table)
PROCESSES = {
    "erosion": ("soilErosionDebt", "CATE_EROSION"),
    "compaction": ("soilCompactionDebt", "CATE_COMPACTION"),
    "soc": ("soilOrganicCarbonDebt", "CATE_SOC"),
    "soil_water": ("soilWaterDebt", "CATE_SOILWATER"),
    "tree_cover": ("treeCoverDebt", "CATE_TREECOVER"),
}

# --------------------------------------------------------------------------- #
# Variables: the forests use the raw (linear-scale) variables
# --------------------------------------------------------------------------- #
OUTCOME = dc.OUTCOME_RAW
TREATMENTS = dc.TREATMENTS_RAW
SOIL_TYPE = dc.CONTROLS_environment_soilType

X_BASE_NUMERIC = (
    dc.CONTROLS_environment_climateWc_RAW
    + dc.CONTROLS_environment_climateTerra_RAW
    + dc.CONTROLS_environment_timeDelta_climateTerra_RAW
    + dc.CONTROLS_environment_terrain_RAW
    + dc.CONTROLS_environment_soilGrid_RAW
    + dc.CONTROLS_adaptation_RAW
    + dc.CONTROLS_socioeconomic_RAW
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def _as_int(s: pd.Series) -> pd.Series:
    """0/1 dummy -> int. Handles numeric columns and pyreadr Categoricals."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = pd.to_numeric(s.astype(str))
    return s.astype(int)


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return (df_ok_noNA, X_base_with_crop_type_controls, Y)."""
    assert_files_exist(dataset_path)
    df_ok = dc.read_rds(dataset_path).reset_index(drop=True)
    df_ok["row_id"] = np.arange(1, len(df_ok) + 1)

    keep = list(dict.fromkeys(
        ["row_id", "long", "lat", OUTCOME, *TREATMENTS,
         "attainableYieldGerberRay", "actualYieldGerberRay",
         *dc.crop_type_dummy_vars, *X_BASE_NUMERIC, SOIL_TYPE]
    ))
    df = df_ok[keep].dropna().reset_index(drop=True)       # tidyr::drop_na()
    for c in dc.crop_type_dummy_vars:
        df[c] = _as_int(df[c])

    df.to_parquet(path_processed("final_dataset", "analysis_ready_data_linear_scale.parquet"))

    # model.matrix(~ soil_type__mode - 1): one indicator per level (full rank, no reference)
    soil_hot = pd.get_dummies(df[SOIL_TYPE].astype("category"), prefix=SOIL_TYPE,
                              prefix_sep="", dtype=float)
    X_base = pd.concat([df[X_BASE_NUMERIC].astype(float), soil_hot], axis=1)
    X_base_crop = pd.concat([X_base, df[dc.crop_type_dummy_vars].astype(float)], axis=1)
    Y = df[OUTCOME].to_numpy(float)
    return df, X_base_crop, Y


# --------------------------------------------------------------------------- #
# Causal forest (grf::causal_forest equivalent)
# --------------------------------------------------------------------------- #
def grf_mtry(p: int) -> int:
    """grf's default mtry = min(ceil(sqrt(p) + 20), p)."""
    return min(math.ceil(math.sqrt(p) + 20), p)


def causal_forest(X: np.ndarray, Y: np.ndarray, W: np.ndarray, *, num_trees: int = 2000,
                  min_node_size: int = 5, num_threads: int = 4, seed: int = 42):
    """Fit a grf-style causal forest. Returns (forest, oob_cate, Y_hat, W_hat).

    grf defaults are mapped as follows:
      sample.fraction = 0.5      -> max_samples = 0.5
      mtry                       -> max_features = grf_mtry(p)
      min.node.size = 5          -> min_samples_leaf = 5
      honesty = TRUE (0.5 split) -> honest = True
      alpha = 0.05               -> min_balancedness_tol = 0.45 (children keep >= 5 %)
      imbalance.penalty = 0      -> (no penalty; default)
      ci.group.size = 2          -> subforest_size = 4 (econml needs n_trees % 4 == 0)
      nuisance forests: max(50, num.trees / 4) trees, OOB predictions
    """
    from econml.grf import CausalForest, RegressionForest

    p = X.shape[1]
    common = dict(
        min_samples_leaf=min_node_size, max_features=grf_mtry(p), max_samples=0.5,
        min_balancedness_tol=0.45, honest=True, n_jobs=num_threads, random_state=seed,
    )

    # 1. Local centering (grf: Y.hat, W.hat from regression forests, OOB)
    n_nuis = max(52, (num_trees // 4) // 4 * 4)  # >= 50 and a multiple of subforest_size
    y_forest = RegressionForest(n_estimators=n_nuis, inference=False, **common).fit(X, Y)
    w_forest = RegressionForest(n_estimators=n_nuis, inference=False, **common).fit(X, W)
    Y_hat = y_forest.oob_predict(X).ravel()
    W_hat = w_forest.oob_predict(X).ravel()

    # 2. Causal forest on the centered data
    n_cf = max(4, num_trees // 4 * 4)
    cf = CausalForest(
        n_estimators=n_cf, criterion="het", fit_intercept=True, inference=True,
        subforest_size=4, **common,
    )
    cf.fit(X, (W - W_hat).reshape(-1, 1), Y - Y_hat)

    # 3. OOB CATE (R: predict(cf)$predictions)
    cate = cf.oob_predict(X).ravel()
    return cf, cate, Y_hat, W_hat


def run_cf_func_customXbase(D_var: str, custom_X_base: pd.DataFrame, df: pd.DataFrame,
                            Y: np.ndarray, out_cf_filename: str, out_df_CATE_filename: str,
                            num_trees: int = 2000, min_node_size: int = 5,
                            num_threads: int = 4) -> pd.DataFrame:
    """Fit a forest for one debt and save the forest and its CATE table.

    X = base controls (+ soil one-hot + crop dummies) + the *other four* debts,
    so each CATE is conditional on the co-occurring degradation processes.
    """
    W = df[D_var].to_numpy(float)
    other_LDs = [t for t in TREATMENTS if t != D_var]
    X = pd.concat([custom_X_base, df[other_LDs].astype(float)], axis=1)

    cf, cate, Y_hat, W_hat = causal_forest(
        X.to_numpy(), Y, W, num_trees=num_trees, min_node_size=min_node_size,
        num_threads=num_threads, seed=42,
    )

    with open(cf_dir / f"{out_cf_filename}.pkl", "wb") as fh:
        pickle.dump({"forest": cf, "X_columns": list(X.columns), "Y_hat": Y_hat, "W_hat": W_hat}, fh)

    df_CATE = pd.DataFrame({"long": df["long"], "lat": df["lat"], "CATE": cate})
    df_CATE.to_csv(cf_dir / f"{out_df_CATE_filename}.csv", index=False)
    df_CATE.to_parquet(cf_dir / f"{out_df_CATE_filename}.parquet", index=False)
    print(f"  {D_var}: n={len(df_CATE)}, mean CATE={cate.mean():.4g}")
    return df_CATE


# --------------------------------------------------------------------------- #
# Rasterization
# --------------------------------------------------------------------------- #
NA_FLAG = -3.4e38


def rasterize_points_max(long: np.ndarray, lat: np.ndarray, values: np.ndarray,
                         transform, shape: tuple[int, int]) -> np.ndarray:
    """Burn point values into a grid, taking the max when several points share a cell
    (R: raster::rasterize(..., fun = "max")). Empty cells are NaN."""
    inv = ~transform
    cols, rows = inv * (np.asarray(long, float), np.asarray(lat, float))
    rows, cols = np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1]) & np.isfinite(values)
    flat = np.full(shape[0] * shape[1], -np.inf)
    np.maximum.at(flat, rows[ok] * shape[1] + cols[ok], np.asarray(values, float)[ok])
    flat[np.isneginf(flat)] = np.nan
    return flat.reshape(shape)


def write_tif(arr: np.ndarray, path: Path, profile: dict) -> None:
    import rasterio

    out = np.where(np.isnan(arr), NA_FLAG, arr).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def rasterize_cates() -> None:
    import rasterio

    cf_raster_dir.mkdir(parents=True, exist_ok=True)
    cate_files = [cf_dir / f"cate_{k}_crop_type_controls.parquet" for k in PROCESSES]
    assert_files_exist([yield_gap_template_path, *cate_files])

    with rasterio.open(yield_gap_template_path) as tmpl:
        transform, shape, crs = tmpl.transform, (tmpl.height, tmpl.width), tmpl.crs

    profile = dict(
        driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="float32",
        crs=crs, transform=transform, nodata=NA_FLAG, compress="deflate",
        tiled=True, blockxsize=512, blockysize=512, num_threads=10,
    )

    # Inner-join the five CATE tables on (long, lat)
    merged = None
    for key, (_, cate_col) in PROCESSES.items():
        t = pd.read_parquet(cf_dir / f"cate_{key}_crop_type_controls.parquet").rename(
            columns={"CATE": cate_col})
        merged = t if merged is None else merged.merge(t, on=["long", "lat"], how="inner")

    total = None
    for key, (_, cate_col) in PROCESSES.items():
        rst = rasterize_points_max(merged["long"], merged["lat"], merged[cate_col].to_numpy(),
                                   transform, shape)
        # keep only positive CATEs (degradation associated with larger yield gaps)
        nonneg = np.where(rst < 0, 0.0, rst)  # NaN stays NaN
        write_tif(nonneg, cf_raster_dir / f"cate_{key}_nonneg.tif", profile)
        total = nonneg if total is None else total + nonneg  # NaN-propagating, as terra '+'

    write_tif(total, cf_raster_dir / "cate_sum_nonneg.tif", profile)
    print(f"  wrote 6 rasters to {cf_raster_dir}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    fit_flag, rast_flag = "LDYG_ALLOW_CAUSAL_FOREST_FIT", "LDYG_ALLOW_CATE_RASTERIZE"
    if not allow_heavy(fit_flag) and not allow_heavy(rast_flag):
        raise RuntimeError(
            "Causal forest fitting and CATE rasterization are long-running stages.\n"
            f"Set {fit_flag}=true to fit forests, or {rast_flag}=true to rasterize saved CATE tables."
        )

    if allow_heavy(fit_flag):
        cf_dir.mkdir(parents=True, exist_ok=True)
        path_processed("final_dataset").mkdir(parents=True, exist_ok=True)
        num_trees = _env_int("LDYG_CF_NUM_TREES", 2000)
        num_threads = _env_int("LDYG_CF_NUM_THREADS", 4)

        df, X_base_with_crop_type_controls, Y = prepare_data()
        # CATE-by-crop inspection needs crop-type-control forests for all five processes
        for key, (D_var, _) in PROCESSES.items():
            run_cf_func_customXbase(
                D_var=D_var,
                custom_X_base=X_base_with_crop_type_controls,
                df=df, Y=Y,
                out_cf_filename=f"causal_forest_{key}_crop_type_controls",
                out_df_CATE_filename=f"cate_{key}_crop_type_controls",
                num_trees=num_trees, num_threads=num_threads,
            )

    if not allow_heavy(rast_flag):
        raise RuntimeError(f"Set {rast_flag}=true to rasterize saved CATE tables.")
    stop_unless_heavy_allowed(rast_flag, "CATE rasterization is a post-fit step.")
    rasterize_cates()


if __name__ == "__main__":
    main()
