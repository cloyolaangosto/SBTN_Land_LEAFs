# `build_cfs_gpkg_from_rasters` File Size Review

This note captures opportunities to reduce the GeoPackage size produced by
`build_cfs_gpkg_from_rasters` in `src/sbtn_leaf/map_calculations.py`.

## 1. Avoid duplicating geometries per flow
The current implementation merges each per-flow result back onto the full master
GeoDataFrame and writes that geometry into the GeoPackage on every iteration.
Each flow therefore re-stores the same geometry rows (`write_df(... append=True)`),
which rapidly inflates file size when hundreds of rasters are processed.

**Recommendation:** Persist the master geometry exactly once (e.g., in a
`master_geometry` layer) and write flow statistics to a separate non-spatial
attribute table keyed by the master identifier. Downstream joins can reattach
the geometry only when needed, but the GeoPackage no longer repeats identical
geometry blobs per flow.

## 2. Remove globally constant columns from the main layer
Within each iteration the code populates `imp_cat` and `unit` with the same
values for all rows. Keeping these string columns for every flow adds redundant
bytes to both the GeoPackage and the CSV export.

**Recommendation:** Move the CF-level metadata (impact category, units, source
file) into a metadata table keyed by `flow_name` or include it once in the CSV
header/GeoPackage layer metadata. The main results layer can store just the
foreign keys and numeric CF statistics.

## 3. Persist the results in a columnar format for CSV output
After concatenating the mean, median, and standard deviation slices, the code
writes a traditional CSV. CSV is row-oriented, uncompressed, and repeats column
names per flow.

**Recommendation:** Consider writing the long-format table to a compressed
columnar format such as Parquet or Feather. These formats apply compression,
preserve data types, and are faster to read/write while taking a fraction of the
space of the equivalent CSV.

## 4. Optional: compress GeoPackage blobs
If keeping a single GeoPackage layer is required, you can still reduce its size
by using GeoPackage tile/feature compression. The `pyogrio.write_dataframe`
function accepts GDAL dataset creation options; setting `GEOMETRY_NAME`,
`SPATIAL_INDEX=NO` (if spatial index is not required), and `COLUMN_TYPES` to
`REAL`/`FLOAT` can reduce overhead. Alternatively, writing to a spatially
indexed Feather/Parquet file and storing geometry separately can offer sizeable
savings when the GeoPackage format is not mandatory.
