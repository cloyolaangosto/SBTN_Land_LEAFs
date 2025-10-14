# Review of `LEAFs/recovered.ipynb`

## Overview
The notebook rebuilds soil erosion LEAF layers by combining RUSLE factors, rasterizing crop-specific C-factors, generating soil erosion rasters, and aggregating results into country-level summaries using helper routines from `sbtn_leaf.map_calculations`.

## Key findings

### 1. Output folder mismatch for soil erosion rasters
* Soil erosion rasters are written using `output_folder = "../LEAFs/soil_erosion/"`, which saves files directly inside `LEAFs/soil_erosion` (no `raster/` subfolder).【F:LEAFs/recovered.ipynb†L994-L999】
* Subsequent cells attempt to open the rasters from `../LEAFs/soil_erosion/raster/`, a directory that does not exist in the repository. This inconsistency will raise `FileNotFoundError` when the notebook is rerun and prevents the later aggregation step from finding any rasters.【F:LEAFs/recovered.ipynb†L1023-L1030】【F:LEAFs/recovered.ipynb†L1592-L1600】

### 2. Incorrect back-transformation of log-normal statistics
* After log-transforming soil erosion data with `np.log1p`, the notebook back-transforms the mean and standard deviation using `np.exp(mean_dt) - 1` and `np.exp(std_dt) - 1`. This is mathematically incorrect for data assumed lognormally distributed; the proper mean should be `np.expm1(mean_dt + 0.5 * std_dt**2)` and the standard deviation requires the full lognormal formula.【F:LEAFs/recovered.ipynb†L1358-L1367】
* Using the incorrect formulas distorts comparisons between the transformed and raw statistics and biases the `x_cut` threshold computed later from these values.【F:LEAFs/recovered.ipynb†L1392-L1397】

### 3. Behavior of helper functions from `map_calculations`
* `mc.multiply_rasters` correctly enforces consistent CRS/shape and carries forward nodata masks when generating the combined R·LS·K product, so its use in the notebook is appropriate.【F:src/sbtn_leaf/map_calculations.py†L1331-L1422】
* `mc.create_binary_mask` assumes the source raster’s nodata value is truthy; if the nodata value were zero, the function would fall into the fallback branch and treat all zero-valued pixels as nodata even when valid. The notebook explicitly passes `src_nodata=255`, so it is safe here, but the helper’s behavior should be noted for other datasets.【F:src/sbtn_leaf/map_calculations.py†L1425-L1495】
* `mc.build_cfs_gpkg_from_rasters` expects rasters to live in the folder passed as `input_folder`, iterating with `os.listdir`. Because the earlier save path omits the `raster/` subdirectory, no rasters will be discovered and the GeoPackage creation will silently operate on an empty list when the notebook is rerun.【F:src/sbtn_leaf/map_calculations.py†L661-L776】【F:LEAFs/recovered.ipynb†L1592-L1600】
* The outlier winsorization requested in the notebook relies on `_apply_outlier_filter` with `method="log1p_win"`, which log-transforms the data, applies the threshold, and caps extreme values before computing weights. This matches the intent described in the notebook for handling skewed erosion rates.【F:src/sbtn_leaf/map_calculations.py†L933-L977】

### 4. Additional observations
* The duplicate assignment of `output_folder` for C-factor rasters is harmless but can be cleaned up.【F:LEAFs/recovered.ipynb†L625-L633】
* Filtering outliers with `data[data <= data_cut]` operates on the masked array returned by Rasterio; the mask is preserved, but if a flattened array was intended the cell should reference `data_f` instead of `data` for clarity.【F:LEAFs/recovered.ipynb†L1386-L1395】

## Recommendations
1. Update the soil erosion export loop to write into a dedicated `../LEAFs/soil_erosion/raster/` directory and ensure that folder exists before aggregation.
2. Replace the back-transformation formulas with the correct lognormal expressions (e.g., `mean_unlogged = np.expm1(mean_dt + 0.5 * std_dt**2)`). Recompute the outlier threshold using the corrected statistics.
3. Consider hardening `mc.create_binary_mask` so that explicit zero-valued nodata settings are handled via equality tests rather than truthiness, preventing misclassification in other datasets.
4. When filtering masked arrays, operate on the flattened `.compressed()` values to avoid ambiguity about masked pixels.
