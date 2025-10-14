# `build_cfs_gpkg_from_rasters` File Size Review

This note captures opportunities to reduce the GeoPackage size produced by
`build_cfs_gpkg_from_rasters` in `src/sbtn_leaf/map_calculations.py`.

## 1. Proposed lightweight export flow

The heaviest part of `build_cfs_gpkg_from_rasters` is the repeated storage of
identical geometries for every raster that is processed. A lightweight
alternative keeps the workflow’s analytical steps intact but limits the
persisted artefacts to:

1. **One master geometry layer** – write the deduplicated
   `master_gdf[[master_key, geometry]]` (plus any essential attributes) a single
   time, e.g. as a `master_geometry` layer inside the GeoPackage.
2. **A long, attribute-only table per raster** – store the per-flow statistics
   (`cf`, `cf_median`, `cf_std`) in a flat file that omits geometry entirely.
   A Parquet or Feather file keeps the table compact and typed; a CSV works too
   if downstream tools do not support columnar formats. When the GeoPackage is
   needed again the long table can be merged with `master_geometry` via
   `master_key`.

This approach trims disk usage because each flow contributes only numeric
columns and a small text identifier instead of a full geometry blob.

If retaining a GeoPackage is optional, the simplest option is to **skip writing
any GeoPackage** and persist only the statistics table. Consumers can later
join the CSV/Parquet back to the `master_shapefile` (or the one-time
`master_geometry` export) inside GIS software. Explicitly documenting that
workflow keeps expectations clear for analysts who receive the lighter package.
`build_cfs_gpkg_from_rasters` now exposes a `write_gpkg` flag so the caller can
toggle the GeoPackage export off while continuing to receive the CSV output.

## 2. Supporting function behaviour

`build_cfs_gpkg_from_rasters` orchestrates several helpers whose behaviour
remains unchanged under the lightweight export strategy:

* `calculate_area_weighted_cfs_from_raster_with_std_and_median_vOutliers`
  reads each raster, reprojects to an equal-area CRS when necessary, and
  computes weighted mean/median/std per region. Its outputs already separate
  tabular attributes (`results_df`) from geometries (`final_gdf`), so the master
  geometry can be stored once while the per-flow table is persisted separately.
* `_fractional_cover_supersample` and `_fractional_cover_exact` control how
  fractional pixel coverage is derived. They only influence numerical
  statistics and do not require geometry duplication.
* `_apply_outlier_filter` and `_weighted_median` provide statistical post-
  processing, producing the scalar values that populate the long table.

Because these helpers all emit attribute data that are already aligned to
`master_key`, no additional refactoring is required when switching the export
mechanism. The main change is the choice of storage format.

## 3. Optional format tweaks

If a GeoPackage must still contain the attribute table, consider enabling GDAL
creation options such as `GEOMETRY_NAME=geom` and `SPATIAL_INDEX=NO` to avoid
extra indexes. When columnar formats are acceptable, favour Parquet/Feather to
obtain compression without writing redundant column headers per flow.
