print("Hello world")


import sbtn_leaf.map_plotting as mp
from sbtn_leaf.paths import data_path, project_path
import geopandas as gpd

raster_test = project_path(
    "LEAFs", "soil_erosion", "raster", "se_rate_25km_Broadleaf_Deciduous_Boreal_dry.tif"
)
world_global_hr = gpd.read_file(
    data_path("world_maps", "high_res", "ne_10m_admin_0_countries.shp")
)

mp.plot_raster_histogram(raster_test)
