# Purpose
This branch is focused on creating new RUSLE LEAFs that have higher resolution utilizing new publications since 2024. The original calculations are based on GloSEM 1.1, with an original pixel resolution of 25km. The objective is to create a new RKL layer at 10 km resolution at least, as the C and P layer are already at a higher resolution.
The R layer depends on rainfall and can change, K and L layer depend on soil properties and terrain properties respectively, and thus will not change dramatically over time and can be assumed as static.

# New Layers
## R Layer (Rainfall erosiditivity)
R layer, depends on rainfall, and could be then adjusted for different future precipitation scenarios based on IPCC, and further predict how much extra erosion from hazards (flooding, drought). This should then remain flexible to be able to run different scenarios.

### Methods
There are different methods described in the papers. These are...

Depending on data availability, the selected is...

## K and L layers
This are constant layers, and thus can be calculated once and be stored as a GeoTiff, they still need to be updated to higher resolution.

### Methods
There are different methods described in the papers. These are...

Depending on data availability, the selected is...

# Incorporating precipitation changes into R layer

# Incorporating Risk Hazards
