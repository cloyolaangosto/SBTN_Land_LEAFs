# LEAF Available
This folder contains all LEAFs generated currently for SBTN Land v2. They are organized into soil quality indicators folder: SOC, soil erosion, and (terrestrial) acidification. 

## Formats available
LEAFs are available in rasters at their respective original resolution in a zip file. For country, sub.country and ecoregions averages, csv tables and geopackages are availalbe for each indicator. Country averages are also available, but should only be selected as last resort as they provide low representation.

Geopackages are organized in two layers: a "geometry_layer", which cointains all geometries as the name indicates, and a soil quality layer, which are specific to each indicator and geographic level of granularity. LEAFs are organized as follow:
| Soil Quality Indicator | Geographic Granulairity | Geopackage LEAF Layer | 
|:---|:---:|:---:|
|SOC|country|soc_leaf_country|
|SOC|subcountry|soc_leaf_subcountry|
|SOC|ecoregion|soc_leaf_ecoregions|
|Soil Erosion|country|se_leaf_country|
|Soil Erosion|subcountry|se_leaf_subcountry|
|Soil Erosion|ecoregion|se_leaf_ecoregions|
|Terrestrial Acidification|country|se_leaf_country|
|Terrestrial Acidification|subcountry|acid_leaf_subcountry|
|Terrestrial Acidification|ecoregion|acid_leaf_ecoregions|

## LEAFs availability and selection
### SOC and Soil Erosion
SOC and soil erosion LEAF are associated with land use and should be selected based on the companies' land use for a given ecoregion or subcountry region.

LEAFs are were calculated for 42 different land use classes, following Morais, Teixeira & Domingos (2019), including 28 agricultural, 15 forest (Needleleaf_Evergreen_Warm_temperate_moist has been eliminated as it's an empty output.), and 1 grassland.

For cereals, 2 residues management options are available (leaving or removing plants residues from the field), 2 tillage option (conventional and reduced tillage) and 2 irrigation options (rainfed or irrigated). For SOC, a total of 8 combinations are thus available, and 4 for soil erosion, as no differentiation is made between irrigation options. 

A total of XXX are available for SOC and YYY for soil erosion. 

#### SOC
Complete documentation on how LEAFs are generated can be found on [SOC Documentation](../documentation/SOC_Documentation.md). Rasters original resolution are $1/12$ of a degree, roughly 9 km at the equator.

SOC LEAFs represent the estimated SOC stock on the land (t SOC/ha).

When using SOC LEAF, bear in mind that they have been generated using Soil Grid's 2016 SOC content as baseline, and extrapolated using RothC model keeping the same land use until 2030. If land use history is available for given plot of land, companies can follow the procedure described in [SOC LEAF Example](../examples/SOC_LEAF_Example.ipynb) to estimate current's SOC and then extrapolate until 2030.

#### Soil Erosion
Complete documentation on how LEAFs are generated can be found on [Soil Erosion Documentation](../documentation/Soil_Erosion_Documentation.md). Rasters original resolution are 25 km.

Soil Erosion LEAFs represent the estimated erosion in one year per hectare (ton soil/ha/year).

### Terrestrial Acidification
Terrestrial Acidification LEAFs are given for 3 acidifying gases $(NO_x, NH_3, SO_x)$, and are also available in raster at XX km resolution, as well as the 3 different geographic level. 

In this case, companies need to match the appropiate LEAF for each gas depending on which area of the world is being emmitted.

These LEAFs represented the potential acidification of the soil for a given gases emmited in a given region when compared to $SO_2$ global average, expressed as $kg SO_2-eq./kg$. 

Lightweight documentation on how these are genearted is provided in [TerrAcidification_Documentation](../documentation/TerrAcidification_Documentation.md)
