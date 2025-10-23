# TODO List

# FIX FOREST AGE HANDLING IN ROTH_C RASTER

## 1) Documentation and Examples
~~1. Write Soil Erosion documentation~~
2. Write Soil Erosion example
3. Write Acidification Example
3.1 Research response option

## 2) High Priority Fixes
### PET Fix
1. Re-run PET-location based and replace.
2. Run PET example again
3. Correct residues distribution
4. Correct soil cover periods. Check below
N. Run all LEAFs again

## 3) New response option LEAFS
1. Run conservative tillage for the same as SOC
2. Re-run conservative tillage with fix PET
3. Run calculations for acidification RO

## 4) Baseline LEAF
1. Run all SOC
~~2. Run all soil erosion~~

# Notes
## Residues fixes
50% on harvest and the rest between the 3 previous months for annual crops and 70% on pruning month and 30% the 4 previous month. 

## Soil Cover period
1 means that the soil was covered with vegetation during that month and 0 means that the soil was uncovered. Soil cover period was set to 1 for permanent crops (orchards, olive groves and vineyards), forests, grasslands and shrublands. For croplands, this parameter was obtained from Chapagain et al. [40] and depends on the thermal zones used by the Global Agro-ecological Zones (GAEZ) Project [41], whose definition is based on temperature and precipitation. They provide sowing/planting dates and the duration of the vegetation cycles. In the months between sowing/plantation and harvesting we considered the soil cover parameter to be 1 and in the other months, in the case where residues are not left on the field and so the soil is bare, we assigned it the value 0. For cereals, when residues are left on the field we used the value 1 for all months.