"""Utility helpers for lazily loading shared crop and climate tables.

The PET and crop calculation modules both rely on the crop coefficient
(`K_Crop_Data.csv`), absolute day (`AbsoluteDayTable.csv`), and climate
lookup tables. Historically each module read these files independently at
import time which duplicated IO and made it awkward to inject custom test
data.

This module centralises the logic for locating those files, loads them on
demand, and caches the parsed ``polars`` objects.  Callers can request a
fresh clone of each table whenever needed or pass the tables around
explicitly for tests.  The consolidated ``_cached_table`` helper coordinates
file discovery, parsing, and cloning so each loader follows the same
pipeline while preserving ``functools.lru_cache`` semantics.
"""

from __future__ import annotations

from functools import lru_cache, partial
from pathlib import Path
import calendar
from typing import Callable, Dict, Iterable, Mapping, Tuple, TypeVar

import geopandas as gpd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

TableT = TypeVar("TableT")


def _ensure_exists(path: Path, *, description: str) -> Path:
    """Ensure ``path`` exists before attempting to read it."""

    if not path.exists():
        raise FileNotFoundError(f"Expected {description} at '{path}'.")
    return path


def _cached_table(
    path_factory: Callable[[], Path],
    reader: Callable[[Path], TableT],
    *,
    description: str,
    clone_method: str,
) -> Tuple[Callable[[], TableT], Callable[[], TableT]]:
    """Construct cached loader/getter pair for tabular resources.

    The returned loader lazily reads from ``path_factory`` on first use while the
    getter clones the cached object each time to protect callers from accidental
    mutation.
    """

    @lru_cache(maxsize=None)
    def load() -> TableT:
        path = _ensure_exists(path_factory(), description=description)
        return reader(path)

    def get() -> TableT:
        table = load()
        return getattr(table, clone_method)()

    return load, get


_load_crop_coefficients_table, get_crop_coefficients_table = _cached_table(
    lambda: DATA_DIR / "crops" / "K_Crop_Data.csv",
    pl.read_csv,
    description="crop coefficient lookup table",
    clone_method="clone",
)
_load_crop_coefficients_table.__doc__ = "Read the crop coefficient CSV from disk."
get_crop_coefficients_table.__doc__ = (
    "Return a cached copy of the crop coefficient table."
)


_load_absolute_day_table, get_absolute_day_table = _cached_table(
    lambda: DATA_DIR / "crops" / "AbsoluteDayTable.csv",
    pl.read_csv,
    description="absolute day lookup table",
    clone_method="clone",
)
_load_absolute_day_table.__doc__ = "Read the absolute day lookup table from disk."
get_absolute_day_table.__doc__ = (
    "Return a cached copy of the absolute day lookup table."
)


@lru_cache(maxsize=None)
def _build_days_in_month_table(year: int = 2023) -> pl.DataFrame:
    """Construct a Polars table containing the number of days per month."""

    return pl.DataFrame(
        {
            "Month": list(range(1, 13)),
            "Days_in_Month": [calendar.monthrange(year, month)[1] for month in range(1, 13)],
        }
    )


def get_days_in_month_table(year: int = 2023) -> pl.DataFrame:
    """Return a cached copy of the days-in-month table for ``year``."""

    return _build_days_in_month_table(year).clone()


_load_crop_naming_index_table, get_crop_naming_index_table = _cached_table(
    lambda: DATA_DIR / "crops" / "crop_naming_index.csv",
    pl.read_csv,
    description="crop naming index table",
    clone_method="clone",
)
_load_crop_naming_index_table.__doc__ = "Read the crop naming index CSV from disk."
get_crop_naming_index_table.__doc__ = (
    "Return a cached copy of the crop naming index table."
)


_load_fao_statistics_table, get_fao_statistics_table = _cached_table(
    lambda: DATA_DIR / "crops" / "Production_Crops_Livestock_E_All_Data.csv",
    pl.read_csv,
    description="FAO production statistics table",
    clone_method="clone",
)
_load_fao_statistics_table.__doc__ = "Read the FAO production statistics CSV from disk."
get_fao_statistics_table.__doc__ = (
    "Return a cached copy of the FAO production statistics table."
)


_load_fao_crop_yields_table, get_fao_crop_yields_table = _cached_table(
    lambda: DATA_DIR / "crops" / "fao_crop_yields_1423.csv",
    partial(pl.read_csv, separator=";"),
    description="FAO crop yields table",
    clone_method="clone",
)
_load_fao_crop_yields_table.__doc__ = "Read the FAO crop yields CSV from disk."
get_fao_crop_yields_table.__doc__ = (
    "Return a cached copy of the FAO crop yields table."
)


_load_country_boundaries, get_country_boundaries = _cached_table(
    lambda: DATA_DIR
    / "CountryLayers"
    / "Country_Level0"
    / "g2015_2014_0.shp",
    gpd.read_file,
    description="country boundary shapefile",
    clone_method="copy",
)
_load_country_boundaries.__doc__ = "Read the country boundary shapefile from disk."
get_country_boundaries.__doc__ = (
    "Return a cached copy of the country boundary shapefile."
)


_load_ecoregions_shapefile, get_ecoregions_shapefile = _cached_table(
    lambda: DATA_DIR / "Ecoregions2017" / "Ecoregions2017.shp",
    gpd.read_file,
    description="ecoregions shapefile",
    clone_method="copy",
)
_load_ecoregions_shapefile.__doc__ = "Read the ecoregions shapefile from disk."
get_ecoregions_shapefile.__doc__ = (
    "Return a cached copy of the ecoregions shapefile."
)


_load_crop_ag_residue_table, get_crop_ag_residue_table = _cached_table(
    lambda: DATA_DIR / "crops" / "crop_residue_data.xlsx",
    partial(pl.read_excel, sheet_name="crop_ABG_Res"),
    description="crop residue workbook",
    clone_method="clone",
)
_load_crop_ag_residue_table.__doc__ = (
    "Read the crop above-ground residue Excel sheet from disk."
)
get_crop_ag_residue_table.__doc__ = (
    "Return a cached copy of the crop above-ground residue table."
)


_load_crop_residue_ratio_table, get_crop_residue_ratio_table = _cached_table(
    lambda: DATA_DIR / "crops" / "crop_residue_data.xlsx",
    partial(pl.read_excel, sheet_name="crop_res_ratios"),
    description="crop residue workbook",
    clone_method="clone",
)
_load_crop_residue_ratio_table.__doc__ = (
    "Read the crop residue ratio Excel sheet from disk."
)
get_crop_residue_ratio_table.__doc__ = (
    "Return a cached copy of the crop residue ratio table."
)


THERMAL_CLIMATE_ROWS = [
    (1, "Tropics, lowland", "Tropics"),
    (2, "Tropics, highland", "Tropics"),
    (3, "Subtropics, summer rainfall", "Subtropics summer rainfall"),
    (4, "Subtropics, winter rainfall", "Subtropics winter rainfall"),
    (5, "Subtropics, low rainfall", "Subtropics winter rainfall"),
    (6, "Temperate, oceanic", "Oceanic temperate"),
    (7, "Temperate, sub-continental", "Sub-continental temperate and continental temperate"),
    (8, "Temperate, continental", "Sub-continental temperate and continental temperate"),
    (9, "Boreal, oceanic", "Sub-continental boreal, continental boreal and polar/arctic"),
    (10, "Boreal, sub-continental", "Sub-continental boreal, continental boreal and polar/arctic"),
    (11, "Boreal, continental", "Sub-continental boreal, continental boreal and polar/arctic"),
    (12, "Arctic", "Sub-continental boreal, continental boreal and polar/arctic"),
]


@lru_cache(maxsize=None)
def _build_thermal_climate_tables() -> Tuple[pl.DataFrame, Dict[int, str], Dict[str, Tuple[int, ...]]]:
    """Create the thermal climate lookup table and related dictionaries."""

    table = pl.DataFrame(THERMAL_CLIMATE_ROWS, schema=["id", "TC_Name", "TC_Group"])

    climate_zone_lookup: Dict[int, str] = dict(zip(table["id"].to_list(), table["TC_Group"].to_list()))

    zone_ids: Dict[str, list[int]] = {}
    for zone_id, group in climate_zone_lookup.items():
        zone_ids.setdefault(group, []).append(zone_id)

    zone_ids_by_group = {group: tuple(ids) for group, ids in zone_ids.items()}

    return table, climate_zone_lookup, zone_ids_by_group


def get_thermal_climate_tables(
    *,
    include_lookup: bool = True,
    include_zone_ids: bool = True,
) -> Tuple[pl.DataFrame, Mapping[int, str], Mapping[str, Iterable[int]]]:
    """Return cached climate tables and associated helper mappings."""

    table, lookup, zone_ids = _build_thermal_climate_tables()

    output_table = table.clone()
    output_lookup: Mapping[int, str]
    output_zone_ids: Mapping[str, Iterable[int]]

    output_lookup = dict(lookup) if include_lookup else {}
    output_zone_ids = dict(zone_ids) if include_zone_ids else {}

    return output_table, output_lookup, output_zone_ids
