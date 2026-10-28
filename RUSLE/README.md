# Purpose
This branch is focused on creating new RUSLE LEAFs that have higher resolution utilizing new publications since 2024. The original calculations are based on GloSEM 1.1, with an original pixel resolution of 25km. The objective is to create a new RKL layer at 10 km resolution at least, as the C and P layer are already at a higher resolution.
The R layer depends on rainfall and can change, K and L layer depend on soil properties and terrain properties respectively, and thus will not change dramatically over time and can be assumed as static.

# New Layers
Three new layers need to be generated: R K and L. Also, three papers are being explored to do it:
1. [Borelli et al. (2022). GloSEM: High-resolution global estimates of present and future soil displacement in croplands by water erosion](https://www.nature.com/articles/s41597-022-01489-x)
2. [Fadl et al. (2025). Integrating RUSLE, AHP, GIS, and cloud-based geospatial analysis for soil erosion assessment under mediterranean conditions](https://www.nature.com/articles/s41598-025-22503-3)
3. [Pyne et al. (2026).Analyzing soil erosion trends and future predictions using the RUSLE model: A case study of the Dwarakeswar River Basin, West Bengal](https://www.sciencedirect.com/science/article/pii/S2095633925000930)

Which method to use will be analyzed, trying to prioritize Borelli et al. (2022) as a previous version of the GloSEM model was used (1.1 instead of the current 1.3), taking into consideration data accessibility, what has already been download, and ease of implementation.

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
