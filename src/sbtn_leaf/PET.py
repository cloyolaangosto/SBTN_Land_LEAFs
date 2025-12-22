# Crop Water Requirement and Evapotranspiration

"""This script calculates the crop water requirement depending on the crop and climate it's located. It's calculated as the simple multiplifcation of potential evapotranspiration (PET) and a crop coefficient depending on it's growing stage (K). PET is calculated using the Thornthwaite equation (originally from Thornthwaite, C. W. An Approach toward a Rational Classification of Climate. Geogr. Rev. 1948, 38, 55, doi:10.2307/210739.), while K is obtained from Chapagain, A. K.; Hoekstra, A. Y. Water footprint of nations. Volume 1 : Main report; 2004."""

## PET - Thornwaite equation
"""This equation calculates PET based on duration of sunlight in hours, varying with season and latitude (L), number of days in a month (N), average monttly air temperature (T, in °C), and heat index (I_a). Detailed explanations can be found here: https://wikifire.wsl.ch/tiki-indexf125.html?page=Potential+evapotranspiration

PET is calculated as:
PET = 0 if T <0,
PET = 1.6 * (L/12) * (N/30) * (10 * T/I)**a

Where a is calcualted as:
a = (6.75 * 10^-7⋅ * I^3)−(7.71*10^−5 * I^2) + (0.01792 * I)+(0.49239)

There has been more updates since it's original publication since 1948,  but to keep consistency with the SOC publication we're replicating, we are using it in it's original format."""


"""
## Crop Coefficient (K)
Crop coefficient is, of course, crop dependent, as well as weather dependent. It is used to adjust the PET_0 that is only weather dependent to crops, depending on their growing stage. It follows a curve, which for annual crops looks like this:

For forage crops, it's different, as they have several rotations throughout a year, and looks like this:

For fruit trees and trees plantation... I need to read more.

Complete documentation can be found here: https://www.fao.org/4/X0490E/x0490e00.htm

The following part of the code tries to automatize this calculations for annual crops, as a first example.
"""

#### Modules ####
import math  # For mathematical operations
from calendar import monthrange  # For getting the number of days in a month
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np  # For numerical operations and array handling
import rasterio  # For raster data handling
from rasterio.warp import reproject, Resampling  # For raster reprojection and resampling
from tqdm import tqdm  # For progress bar

import polars as pl  # For data manipulation and analysis

from sbtn_leaf.paths import data_path

PathLike = Union[str, Path]


def _as_path(value: PathLike) -> Path:
    """Return ``value`` as a :class:`~pathlib.Path` instance."""

    return value if isinstance(value, Path) else Path(value)


def _resolve_raster_path(path: PathLike) -> Path:
    """Resolve ``path`` to the appropriate on-disk raster location."""

    candidate = _as_path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate

    data_candidate = data_path(candidate)
    if data_candidate.exists():
        return data_candidate

    return data_candidate


_DEFAULT_PET_BASE_RASTER_PATH = data_path("soil_weather", "uhth_pet_locationonly.tif")
_DEFAULT_THERMAL_ZONE_RASTER_PATH = data_path("soil_weather", "uhth_thermal_climates.tif")

from sbtn_leaf.data_loader import (
    get_absolute_day_table,
    get_crop_coefficients_table,
    get_days_in_month_table,
    get_thermal_climate_tables,
)


###################
#### Functions ####
###################


def _resolve_crop_table(crop_table: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """Return the provided crop table or fetch the shared cached version."""

    if crop_table is not None:
        return crop_table
    return get_crop_coefficients_table()


def _resolve_abs_date_table(abs_date_table: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """Return the provided absolute day table or fetch the shared cached version."""

    if abs_date_table is not None:
        return abs_date_table
    return get_absolute_day_table()


def _resolve_days_in_month_table(
    days_table: Optional[pl.DataFrame] = None,
    *,
    year: int = 2023,
) -> pl.DataFrame:
    """Return the provided days-in-month table or fetch the shared cached version."""

    if days_table is not None:
        return days_table
    return get_days_in_month_table(year)


def _resolve_zone_mappings(
    zone_ids_by_group: Optional[Mapping[str, Iterable[int]]] = None,
    climate_zone_lookup: Optional[Mapping[int, str]] = None,
    climate_table: Optional[pl.DataFrame] = None,
):
    """Return thermal climate helpers, pulling cached defaults when omitted."""

    if all(value is not None for value in (zone_ids_by_group, climate_zone_lookup, climate_table)):
        return climate_table, climate_zone_lookup, zone_ids_by_group

    table, lookup, zones = get_thermal_climate_tables()
    return (
        climate_table or table,
        climate_zone_lookup or lookup,
        zone_ids_by_group or zones,
    )


def _normalize_landuse_mask(mask: np.ndarray, *, expected_shape: Tuple[int, int]) -> np.ndarray:
    """Return a boolean land-use mask, validating its dimensionality."""

    mask_array = np.asarray(mask)
    if mask_array.shape != expected_shape:
        raise ValueError(
            "landuse_mask must match raster dimensions "
            f"{expected_shape}; got {mask_array.shape}."
        )

    if mask_array.dtype == bool:
        return mask_array

    return mask_array.astype(int) == 1


def _load_pet_inputs(
    pet_base_raster_path: PathLike,
    thermal_zone_raster_path: PathLike,
    *,
    landuse_raster_path: Optional[PathLike] = None,
    landuse_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load the PET base, thermal zones, and land-use mask arrays.

    Parameters
    ----------
    pet_base_raster_path:
        Path to the PET base raster containing the 12 monthly bands.
    thermal_zone_raster_path:
        Path to the raster encoding the thermal zone identifiers.
    landuse_raster_path:
        Optional path to a land-use raster aligned with the PET base and
        thermal rasters. Mutually exclusive with ``landuse_mask``.
    landuse_mask:
        Optional 2D array providing the land-use mask in-memory. Mutually
        exclusive with ``landuse_raster_path``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]
        The PET base array (12, H, W), the thermal zone array (H, W), a
        boolean land-use mask (H, W), and the raster profile copied from the
        PET base dataset.
    """

    strategy_count = int(landuse_raster_path is not None) + int(landuse_mask is not None)
    if strategy_count != 1:
        raise ValueError(
            "Exactly one of landuse_raster_path or landuse_mask must be provided."
        )

    pet_base_path = _resolve_raster_path(pet_base_raster_path)
    thermal_zone_path = _resolve_raster_path(thermal_zone_raster_path)

    with rasterio.open(pet_base_path) as pet_ds, rasterio.open(
        thermal_zone_path
    ) as thermal_ds:
        if (
            pet_ds.crs != thermal_ds.crs
            or pet_ds.transform != thermal_ds.transform
            or pet_ds.width != thermal_ds.width
            or pet_ds.height != thermal_ds.height
        ):
            raise ValueError("CRS/transform/shape mismatch between PET and thermal rasters")

        pet_base = pet_ds.read(out_dtype="float32", masked=True)
        if np.ma.isMaskedArray(pet_base):
            pet_base = pet_base.filled(np.float32(np.nan))
        else:
            pet_base = pet_base.astype("float32", copy=False)

        thermal_zones = thermal_ds.read(1)
        if np.ma.isMaskedArray(thermal_zones):
            thermal_zones = thermal_zones.filled(0)
        thermal_zones = thermal_zones.astype(int)

        expected_shape = (pet_ds.height, pet_ds.width)

        if landuse_raster_path is not None:
            landuse_path = _resolve_raster_path(landuse_raster_path)
            with rasterio.open(landuse_path) as landuse_ds:
                if (
                    pet_ds.crs != landuse_ds.crs
                    or pet_ds.transform != landuse_ds.transform
                    or pet_ds.width != landuse_ds.width
                    or pet_ds.height != landuse_ds.height
                ):
                    raise ValueError("CRS/transform/shape mismatch between PET and land-use rasters")

                landuse_data = landuse_ds.read(1)
                if np.ma.isMaskedArray(landuse_data):
                    landuse_data = landuse_data.filled(0)
                landuse_mask_array = landuse_data.astype(int) == 1
        else:
            landuse_mask_array = _normalize_landuse_mask(
                landuse_mask, expected_shape=expected_shape
            )

        profile = pet_ds.profile.copy()

    return pet_base, thermal_zones, landuse_mask_array, profile

def get_month_from_absolute_day(abs_day: int, abs_date_table: Optional[pl.DataFrame] = None):
    """
    Get the month from an absolute day number.

    Parameters:
    - abs_day: int, absolute day number (1 to 365)

    Returns:
    - int, month number (1 = January, ..., 12 = December)
    """
    if abs_day < 1 or abs_day > 365:
        raise ValueError("Absolute day must be between 1 and 365.")

    abs_table = _resolve_abs_date_table(abs_date_table)

    # Find the month by iterating through the days in each month
    month = abs_table.filter(pl.col('Day_Num') == abs_day).select('Month').item()

    return month

##### Potential Evapotranspiration (PET) #####
def daylight_duration(latitude_deg, month):
    """
    Estimate daylight duration in hours based on latitude and month.

    Parameters:
    - latitude_deg: float, latitude in degrees (-90 to 90)
    - month: int, month number (1 = January, ..., 12 = December)

    Returns:
    - Daylight duration in hours (approximate)
    """
    # Convert latitude to radians
    lat_rad = math.radians(latitude_deg)

    # Approximate solar declination angle δ in radians (Cooper’s formula)
    day_of_year = [15, 45, 74, 105, 135, 162, 198, 228, 258, 288, 318, 344]  # mid-month days
    n = day_of_year[month - 1]
    decl_rad = math.radians(23.44) * math.sin(math.radians((360 / 365.0) * (n - 81)))

    # Calculate hour angle ω₀
    cos_omega = -math.tan(lat_rad) * math.tan(decl_rad)
    if cos_omega >= 1:
        # Polar night (no sunrise)
        return 0.0
    elif cos_omega <= -1:
        # Midnight sun (24h daylight)
        return 24.0
    else:
        omega = math.acos(cos_omega)
        return (2 * omega * 180 / math.pi) / 15  # convert radians to hours (15° per hour)
    

def calcualte_heat_index(air_temp: float):
    """
    This is calcualted as: I = (T/5)^1.514
    """
    
    heat_index = (max(0, air_temp)/5)**1.514

    return heat_index

def calcualte_a(heat_index: float):
    exponent = (6.75 * 10**(-7) * heat_index**3) - (7.71 * 10**(-5) * heat_index**2) + (0.01792 * heat_index) + 0.49239
    
    return exponent


def calculate_PET_location_based(monthly_temps, year: int, lat: float):
    """
    Calculate the Potential Evapotranspiration (PET) in mm per month using the Thornthwaite equation. It doesn't take into consideration the crop growing, only the location and the month.

    Parameters:
    - monthly_temps: air temperature series in degrees Celsius
    - year: year
    - lat: float, latitude in degrees

    Returns:
    - PET series in mm per month
    """
    # Checks there are 12 months of temp
    if len(monthly_temps) != 12:
        raise ValueError("monthly_temps must contain exactly 12 values for each month.")

    # Step 1: Calculate I (heat index)
    I = sum(calcualte_heat_index(temp) for temp in monthly_temps)

    # Step 2: Calculate exponent a
    a  = calcualte_a(I)
    
    # Step 3: Calculate PET Series 
    PET = []
    for i, T in enumerate(monthly_temps):
        if T <= 0:
            PET.append(0.0)
            continue

        # Days in the month
        N = monthrange(year, i+1)[1]  # 2024 is leap year-safe

        # Day length
        L = daylight_duration(lat, i + 1)

        # Thornthwaite formula
        PET_i = 16 * (L / 12) * (N / 30) * ((10 * T / I) ** a)
        PET.append(PET_i)

    return PET


##### Crop Coefficient (K) Functions #####
def correct_abs_date(num_day:int):
    if num_day > 365:
        num_day = num_day - 365
    
    return num_day

def create_KC_Curve(
    crop: str,
    climate: str,
    *,
    crop_table: Optional[pl.DataFrame] = None,
    abs_date_table: Optional[pl.DataFrame] = None,
):
    crop_table = _resolve_crop_table(crop_table)
    abs_table = _resolve_abs_date_table(abs_date_table)

    # First, check that both crops and climate are in the K_Crops table
    if crop not in crop_table['Crop'].unique() or climate not in crop_table['Climate_Zone'].unique():
        raise ValueError("Specified crop or climate not found in Kc data.")
    # else:
    #    print(f"Creating Kc curve for crop: {crop} in climate: {climate}")
    
    # Retrieve data for the specified crop and climate
    crop_data = crop_table.filter((pl.col('Crop') == crop) & (pl.col('Climate_Zone') == climate))
    # print("Crop data retrieved:")
    # print(crop_data.columns)

    # Stage dataframe for 365 days
    Kc_df = pl.DataFrame({
        "day_num": range(1, 366),
        "Stage_id": [0] * 365,
        "Kc": [0.0] * 365
    })

    # Calculating start and end dates for each phase
    KC_Dates = pl.DataFrame({
        "Stage": ["Planting", "Initial", "Development", "Mid", "Late"],
        "Stage_id": [1, 2, 3, 4, 5],
        "Start_Day": [0, 0, 0, 0, 0],
        "End_Day": [0, 0, 0, 0, 0]
    })

    # print("Retrieving stages dates for crop:", crop)
    
    # Get the phanse day number as an integer value
    planting_day_num_start = abs_table.filter(
        pl.col('Date') == crop_data['Planting_Greenup_Date']).select('Day_Num').item()
    planting_day_num_end = planting_day_num_start
    initial_stage_duration = int(crop_data['Initial_days'][0])
    dev_stage_duration = int(crop_data['Dev_days'][0])
    mid_stage_duration = int(crop_data['Mid_days'][0])
    late_stage_duration = int(crop_data['Late_days'][0])

    def _stage_end(start_day: int, duration: int) -> int:
        return start_day + duration - 1 if duration > 0 else start_day - 1

    Initial_day_start = planting_day_num_end + 1
    Initial_day_end = _stage_end(Initial_day_start, initial_stage_duration)
    Development_day_start = Initial_day_end + 1
    Development_day_end = _stage_end(Development_day_start, dev_stage_duration)
    Mid_day_start = Development_day_end + 1
    Mid_day_end = _stage_end(Mid_day_start, mid_stage_duration)
    Late_day_start = Mid_day_end + 1
    Late_day_end = _stage_end(Late_day_start, late_stage_duration)

    # Correct absolute day numbers if they're above 365
    Initial_cday_start = correct_abs_date(Initial_day_start)
    Initial_cday_end = correct_abs_date(Initial_day_end)
    Development_cday_start = correct_abs_date(Development_day_start)    
    Development_cday_end = correct_abs_date(Development_day_end)
    Mid_cday_start = correct_abs_date(Mid_day_start)
    Mid_cday_end = correct_abs_date(Mid_day_end)
    Late_cday_start = correct_abs_date(Late_day_start)
    Late_cday_end = correct_abs_date(Late_day_end)

    # Filling planting date
    KC_Dates = KC_Dates.with_columns(
        pl.when(pl.col('Stage') == 'Planting').then(planting_day_num_start).when(pl.col('Stage') == 'Initial').then(Initial_cday_start).when(pl.col('Stage') == 'Development').then(Development_cday_start).when(pl.col('Stage') == 'Mid').then(Mid_cday_start).otherwise(Late_cday_start).alias('Start_Day'),
        pl.when(pl.col('Stage') == 'Planting').then(planting_day_num_end).when(pl.col('Stage') == 'Initial').then(Initial_cday_end).when(pl.col('Stage') == 'Development').then(Development_cday_end).when(pl.col('Stage') == 'Mid').then(Mid_cday_end).otherwise(Late_cday_end).alias('End_Day'),
    )

    #print("Stages dates:")
    #print(KC_Dates)
    
    # Filling Kc values
    # print("Assigning Kc values to stages")

    stage_records = [
        {
            "day_num": planting_day_num_start,
            "Kc": crop_data['K_ini'][0],
            "Stage_id": 1,
        }
    ]

    initial_days = [
        correct_abs_date(Initial_day_start + offset)
        for offset in range(initial_stage_duration)
    ]
    stage_records.extend(
        {
            "day_num": day,
            "Kc": crop_data['K_ini'][0],
            "Stage_id": 2,
        }
        for day in initial_days
    )

    dev_days = [
        correct_abs_date(Development_day_start + offset)
        for offset in range(dev_stage_duration)
    ]
    if dev_stage_duration > 0:
        if dev_stage_duration == 1:
            dev_values = [crop_data['K_mid'][0]]
        else:
            Kc_dev_slope = (crop_data['K_mid'][0] - crop_data['K_ini'][0]) / (dev_stage_duration - 1)
            dev_values = [
                crop_data['K_ini'][0] + Kc_dev_slope * offset
                for offset in range(dev_stage_duration)
            ]
        stage_records.extend(
            {
                "day_num": day,
                "Kc": value,
                "Stage_id": 3,
            }
            for day, value in zip(dev_days, dev_values)
        )

    mid_days = [
        correct_abs_date(Mid_day_start + offset)
        for offset in range(mid_stage_duration)
    ]
    stage_records.extend(
        {
            "day_num": day,
            "Kc": crop_data['K_mid'][0],
            "Stage_id": 4,
        }
        for day in mid_days
    )

    late_days = [
        correct_abs_date(Late_day_start + offset)
        for offset in range(late_stage_duration)
    ]
    if late_stage_duration > 0:
        if late_stage_duration == 1:
            late_values = [crop_data['K_Late'][0]]
        else:
            Kc_late_slope = (crop_data['K_Late'][0] - crop_data['K_mid'][0]) / (late_stage_duration - 1)
            late_values = [
                crop_data['K_mid'][0] + Kc_late_slope * offset
                for offset in range(late_stage_duration)
            ]
        stage_records.extend(
            {
                "day_num": day,
                "Kc": value,
                "Stage_id": 5,
            }
            for day, value in zip(late_days, late_values)
        )

    stage_assignments = pl.DataFrame(stage_records)
    if not stage_assignments.is_empty():
        Kc_df = Kc_df.join(stage_assignments, on="day_num", how="left", suffix="_stage")
        Kc_df = Kc_df.with_columns(
            pl.when(pl.col("Kc_stage").is_not_null())
            .then(pl.col("Kc_stage"))
            .otherwise(pl.col("Kc"))
            .alias("Kc"),
            pl.when(pl.col("Stage_id_stage").is_not_null())
            .then(pl.col("Stage_id_stage"))
            .otherwise(pl.col("Stage_id"))
            .alias("Stage_id"),
        ).drop(["Kc_stage", "Stage_id_stage"])

    # Kc_df =Kc_df.join(KC_Dates.select('Stage', 'Stage_id'), on="Stage_id", how="left")
    # print("Kc curve created successfully for crop:", crop, "in climate:", climate)
    
    return Kc_df

def monthly_KC_curve(
    crop: str,
    climate: str,
    *,
    crop_table: Optional[pl.DataFrame] = None,
    abs_date_table: Optional[pl.DataFrame] = None,
):
    daily_Kc_curve = create_KC_Curve(
        crop,
        climate,
        crop_table=crop_table,
        abs_date_table=abs_date_table,
    )

    abs_table = _resolve_abs_date_table(abs_date_table)

    # Group by month and calculate average Kc for each month
    absday_month = abs_table.select(
        pl.col('Day_Num'),
        pl.col('Month')
    )

    monthly_Kc = daily_Kc_curve.join(absday_month, left_on='day_num', right_on='Day_Num', how='left').group_by('Month').agg(
        pl.col('Kc').mean().alias('Kc')
    ).sort('Month')

    return monthly_Kc


def _build_monthly_kc_vectors(
    crop_name: str,
    zone_groups: Mapping[str, Iterable[int]],
    crop_table: pl.DataFrame,
    abs_table: pl.DataFrame
):
    """Return a mapping of zone group names to monthly Kc vectors.

    Parameters
    ----------
    crop_name:
        Name of the crop whose Kc curves should be generated.
    zone_groups:
        Mapping from zone group identifier to the collection of thermal zone ids
        associated with that group.
    crop_table:
        Crop coefficient table to use when building the Kc curves.
    abs_table:
        Absolute day lookup table providing month assignments for each day of the
        year.
    tqdm_desc:
        Optional description passed to :func:`tqdm.tqdm` so caller-specific log
        messages remain intact.
    log_template:
        Optional per-group log message that will be formatted with ``group``.
    """

    unique_groups = list(zone_groups)
    kc_by_group = {}
    for group in unique_groups:
        kc_df = monthly_KC_curve(
            crop_name,
            group,
            crop_table=crop_table,
            abs_date_table=abs_table,
        )
        kc_by_group[group] = kc_df.sort("Month")["Kc"].to_numpy()

    return kc_by_group


def calculate_PET_crop_based(
    crop: str,
    climate_zone: str,
    monthly_temps,
    year: int,
    lat: float,
    *,
    crop_table: Optional[pl.DataFrame] = None,
    abs_date_table: Optional[pl.DataFrame] = None,
    days_in_month_table: Optional[pl.DataFrame] = None,
):
    """
    Calculate crop-adjusted Potential Evapotranspiration (PET) on a daily, monthly, and annual basis.

    This function uses crop-specific crop coefficient (Kc) curves and location-based monthly PET
    estimates to derive daily PET values for a full calendar year, accounting for crop phenology
    and climate zone.

    Parameters
    ----------
    crop : str
        Name of the crop for which to calculate PET (must exist in the K_Crops table).
    climate_zone : str
        Climate zone associated with the crop (must match entries in the K_Crops table).
    monthly_temps : list or array-like
        Monthly average temperatures for the location (length 12).
    year : int
        Year used for determining leap years and calendar structure.
    lat : float
        Latitude of the location (in decimal degrees, used for PET calculation).

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'PET_Annual' : float
            Total annual PET (mm/year) for the specified crop and location.
        - 'PET_Monthly' : pl.DataFrame
            Monthly PET values (mm/month), grouped by calendar month.
        - 'PET_Daily' : pl.DataFrame
            Daily PET values (mm/day), including day of year, crop stage, and stage Kc.

    Notes
    -----
    - This function accepts optional data tables for easier testing. When not
      provided, shared cached tables from :mod:`sbtn_leaf.data_loader` are used.
    - Monthly PET is computed from Kc-adjusted daily values.
    - Leap years are handled automatically through day correction logic.
    """

    crop_table = _resolve_crop_table(crop_table)
    abs_table = _resolve_abs_date_table(abs_date_table)
    days_table = _resolve_days_in_month_table(days_in_month_table, year=year)

    # Create Kc curve for the specified crop and climate zone
    Kc_curve = create_KC_Curve(
        crop,
        climate_zone,
        crop_table=crop_table,
        abs_date_table=abs_table,
    )

    # Calculate PET using the Kc values from the curve and monthly temperatures
    PET0 = calculate_PET_location_based(monthly_temps, year, lat)
    PET0 = pl.DataFrame({'Month': range(1, 13), 'PET0_Month': PET0})

    PET_daily = Kc_curve
    PET_daily = PET_daily.join(abs_table, left_on='day_num', right_on='Day_Num', how='left')
    PET_daily = PET_daily.join(PET0, on='Month', how='left').join(days_table, on='Month', how='left')
    PET_daily = PET_daily.with_columns(
        PET_Daily = pl.col('Kc') * pl.col('PET0_Month')/pl.col("Days_in_Month")# Convert monthly PET to daily PET
    )

    PET_Monthly = PET_daily.group_by('Month').agg(
        pl.col('PET_Daily').sum().alias('PET_Monthly')
    ).sort('Month')

    PET_Annual = PET_Monthly['PET_Monthly'].sum()

    results = {
        'PET_Annual': PET_Annual,
        'PET_Monthly': PET_Monthly,
        'PET_Daily': PET_daily
    }

    print("PET calculation completed successfully for crop:", crop, "in climate zone:", climate_zone)
    return results


def _calculate_crop_based_pet_core(
    crop_name: str,
    pet_base: np.ndarray,
    thermal_zones: np.ndarray,
    *,
    zone_groups: Mapping[str, Iterable[int]],
    crop_table: pl.DataFrame,
    abs_table: pl.DataFrame,
    landuse_mask: Optional[np.ndarray] = None,
    compute_annual: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply crop coefficients to a base PET raster for each thermal zone group.

    Parameters
    ----------
    crop_name:
        Name of the crop whose Kc curves should be generated.
    pet_base:
        ``(12, H, W)`` array containing the base PET values that will be adjusted
        by the crop coefficients.
    thermal_zones:
        ``(H, W)`` array of thermal zone identifiers that align with
        ``pet_base``.
    zone_groups:
        Mapping from zone group to the collection of thermal zone identifiers
        associated with that group.
    crop_table, abs_table:
        Lookup tables required to generate the crop coefficient curves.
    landuse_mask:
        Optional boolean mask identifying which pixels should receive crop PET
        values. When omitted, all pixels are considered valid.
    compute_annual:
        When ``True`` the function also aggregates monthly values into an
        annual array.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        The monthly PET array and, when requested, the annual PET array.
    """

    if pet_base.shape[0] != 12:
        raise ValueError(
            "pet_base is expected to contain 12 monthly bands; "
            f"got shape {pet_base.shape}."
        )

    if pet_base.shape[1:] != thermal_zones.shape:
        raise ValueError(
            "pet_base and thermal_zones must share the same spatial dimensions; "
            f"got {pet_base.shape[1:]} and {thermal_zones.shape}."
        )

    if landuse_mask is None:
        landuse_mask = np.ones_like(thermal_zones, dtype=bool)
    else:
        if landuse_mask.shape != thermal_zones.shape:
            raise ValueError(
                "landuse_mask must match the spatial dimensions of thermal_zones; "
                f"got {landuse_mask.shape} and {thermal_zones.shape}."
            )
        landuse_mask = landuse_mask.astype(bool, copy=False)

    pet_monthly = np.full_like(pet_base, np.nan, dtype="float32")
    pet_annual: Optional[np.ndarray]
    if compute_annual:
        pet_annual = np.full(thermal_zones.shape, np.nan, dtype="float32")
    else:
        pet_annual = None

    kc_by_group = _build_monthly_kc_vectors(
        crop_name,
        zone_groups,
        crop_table,
        abs_table
    )

    for group, kc_vec in kc_by_group.items():
        valid_zones = zone_groups[group]
        valid_zones_array = np.array(valid_zones)
        mask = np.isin(thermal_zones, valid_zones_array) & landuse_mask
        if not mask.any():
            continue

        monthly_crop = pet_base[:, mask] * kc_vec[:, None]
        pet_monthly[:, mask] = monthly_crop

        if pet_annual is not None:
            annual_values = np.asarray(np.nansum(monthly_crop, axis=0), dtype="float32")
            all_nan = np.all(np.isnan(monthly_crop), axis=0)
            annual_values[all_nan] = np.float32(np.nan)
            pet_annual[mask] = annual_values

    return pet_monthly, pet_annual


def calculate_crop_based_PET_raster(
    crop_name: str,
    output_monthly_path: PathLike,
    *,
    output_annual_path: Optional[PathLike] = None,
    landuse_raster_path: Optional[PathLike] = None,
    landuse_mask: Optional[np.ndarray] = None,
    pet_base_raster_path: PathLike = _DEFAULT_PET_BASE_RASTER_PATH,
    thermal_zone_raster_path: PathLike = _DEFAULT_THERMAL_ZONE_RASTER_PATH,
    crop_table: Optional[pl.DataFrame] = None,
    abs_date_table: Optional[pl.DataFrame] = None,
    zone_ids_by_group: Optional[Mapping[str, Iterable[int]]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Calculate crop-specific PET rasters using a unified entry point.

    Exactly one land-use input strategy must be provided via ``landuse_raster_path``
    or ``landuse_mask``. The function writes the monthly PET raster to
    ``output_monthly_path`` and, when ``output_annual_path`` is supplied, also
    writes the annual PET raster. The in-memory monthly (and optionally annual)
    arrays are returned for further processing.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        The monthly PET raster (12, H, W) and, when requested, the annual PET
        raster (H, W).
    """

    crop_table = _resolve_crop_table(crop_table)
    abs_table = _resolve_abs_date_table(abs_date_table)
    _, _, zone_groups = _resolve_zone_mappings(zone_ids_by_group=zone_ids_by_group)
    if zone_groups is None:
        zone_groups = {}

    if crop_name not in crop_table["Crop"].unique():
        raise ValueError(f"Crop '{crop_name}' not found in K_Crops table.")

    output_monthly_path = _as_path(output_monthly_path)
    output_annual_path = _as_path(output_annual_path) if output_annual_path is not None else None

    pet_base, thermal_zones, resolved_landuse_mask, profile = _load_pet_inputs(
        pet_base_raster_path,
        thermal_zone_raster_path,
        landuse_raster_path=landuse_raster_path,
        landuse_mask=landuse_mask,
    )

    compute_annual = output_annual_path is not None
    pet_monthly, pet_annual = _calculate_crop_based_pet_core(
        crop_name,
        pet_base,
        thermal_zones,
        zone_groups=zone_groups,
        crop_table=crop_table,
        abs_table=abs_table,
        landuse_mask=resolved_landuse_mask,
        compute_annual=compute_annual,
    )

    print(f"PET calculation completed for crop '{crop_name}' succesfully.")

    monthly_profile = profile.copy()
    monthly_profile.update(count=12, dtype="float32", nodata=np.nan)
    with rasterio.open(output_monthly_path, "w", **monthly_profile) as dst:
        dst.write(pet_monthly)

    if compute_annual and pet_annual is not None and output_annual_path is not None:
        annual_profile = profile.copy()
        annual_profile.update(count=1, dtype="float32", nodata=np.nan)
        with rasterio.open(output_annual_path, "w", **annual_profile) as dst:
            dst.write(pet_annual, 1)

    return pet_monthly, pet_annual


def calculate_crop_based_PET_raster_optimized(
    crop_name: str,
    landuse_raster_path: PathLike,
    output_monthly_path: PathLike,
    output_annual_path: PathLike,
    pet_base_raster_path: PathLike = _DEFAULT_PET_BASE_RASTER_PATH,
    thermal_zone_raster_path: PathLike = _DEFAULT_THERMAL_ZONE_RASTER_PATH,
    *,
    crop_table: Optional[pl.DataFrame] = None,
    abs_date_table: Optional[pl.DataFrame] = None,
    zone_ids_by_group: Optional[Mapping[str, Iterable[int]]] = None,
):
    """Deprecated wrapper for :func:`calculate_crop_based_PET_raster`.

    This function preserves the original interface while delegating the work to
    :func:`calculate_crop_based_PET_raster`. New code should call the unified
    entry point directly.
    """

    calculate_crop_based_PET_raster(
        crop_name,
        output_monthly_path,
        output_annual_path=output_annual_path,
        landuse_raster_path=landuse_raster_path,
        pet_base_raster_path=pet_base_raster_path,
        thermal_zone_raster_path=thermal_zone_raster_path,
        crop_table=crop_table,
        abs_date_table=abs_date_table,
        zone_ids_by_group=zone_ids_by_group,
    )



def calculate_crop_based_PET_raster_vPipeline(
    crop_name: str,
    landuse_array: np.ndarray,
    output_monthly_path: PathLike,
    pet_base_raster_path: PathLike = _DEFAULT_PET_BASE_RASTER_PATH,
    thermal_zone_raster_path: PathLike = _DEFAULT_THERMAL_ZONE_RASTER_PATH,
    *,
    crop_table: Optional[pl.DataFrame] = None,
    abs_date_table: Optional[pl.DataFrame] = None,
    zone_ids_by_group: Optional[Mapping[str, Iterable[int]]] = None,
):
    """Deprecated wrapper for :func:`calculate_crop_based_PET_raster`.

    The ``landuse_array`` argument is forwarded as the in-memory land-use mask.
    New code should call the unified entry point directly.
    """

    pet_monthly, _ = calculate_crop_based_PET_raster(
        crop_name,
        output_monthly_path,
        landuse_mask=landuse_array,
        pet_base_raster_path=pet_base_raster_path,
        thermal_zone_raster_path=thermal_zone_raster_path,
        crop_table=crop_table,
        abs_date_table=abs_date_table,
        zone_ids_by_group=zone_ids_by_group,
    )

    return pet_monthly
