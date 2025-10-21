# Terrestrial Acidification Documentation
Terrestrial Acidification LEAFs are obtained directly from [Roy et al. (2014)](https://www.sciencedirect.com/science/article/abs/pii/S0048969714012789?via%3Dihub), with no modelling performed and only regional and global averages calculated, correcting some calculations errors from the original publication, as has also been described in the [Impact World Plus v2.0 methods](https://www.impactworldplus.org/version-2-0/).

## Step 1 - Data Gathering and Processing
Original shapefiles from Roy et al. (2014) were obtained directly from authors, expressed on a grid of $2° \times 2.5°$ resolution. Shapefiles contained soil sensitivity factors multiplied by fate factors $\textstyle (FF \times SF)$, being expressed in mol H+ L−1 × m2 × kgem−1 × yr, and thus need to be converted into $\textstyle SO2-eq./kg_{emitted}$. Shared files also provided annual emissions of $\textstyle NH_3$, $\textstyle NO_x$, and $\textstyle SO_2$ per pixel. To calculate terrestrial acidification LEAFs, the following process was applied:
1. Merge gases emissions into shapefile based on "CELL_ID"
2. Calculate $\textstyle SO_2$ acidification potential by multiplying $\textstyle (FF \times SF)$ cell value by their emission
3. Calculate total $\textstyle SO_2$ emissions and acidification potential (AP). 
4. Calculate a normalization factor as $\textstyle {NF}_{SO_2} = AP_{SO_2}/(Total {Emissions {SO_2}})$
5. Calculate each gas acidification potential for each cell as $\textstyle AP_{cell,gas}^{SO_2} = AP_{cell,gas}^{mol H^+}/{NF}_{SO_2}$, which finally renders the acidification potential in $\textstyle SO2-eq./kg_{emitted}$

## Step 2 - Data Harmonization
After calculating the acidification potential for each gas in the shapefile, this was transformed into the standard UHTH raster grid used as a base for all LEAFs calculation.

## Step 3 - LEAFs Averages for Ecoregions, Countries and Subcountries
After harmonizing the LEAFs, averages per country, subcountry and ecoregions are calculated. No outlier filtering method is needed due to the factors distribution and coverage.