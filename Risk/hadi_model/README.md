# README for "Land degradation is associated with larger crop yield gaps across global croplands"

## Article Information

**Title:** Land degradation is associated with larger crop yield gaps across global croplands

**Journal status:** Forthcoming in *Nature Food*.

**Authors:** Hadi Hadi<sup>1,*</sup>, D. K. Ray<sup>2</sup>, P. Borrelli<sup>3,4</sup>, J. S. Gerber<sup>2,5</sup>, N. Shokri<sup>6,7</sup>, S. Roman<sup>1</sup>, D. Wuepper<sup>1</sup>

**Affiliations:**

1. Land Economics Group, University of Bonn, 53115 Bonn, Germany
2. Institute on the Environment, University of Minnesota, Saint Paul, MN 55108, United States
3. Department of Environmental Sciences, Environmental Geosciences, University of Basel, 4056 Basel, Switzerland
4. Department of Science, Roma Tre University, 00146 Rome, Italy
5. Project Drawdown, Saint Paul, MN 55101, United States
6. Institute of Geo-Hydroinformatics, Hamburg University of Technology, 21073 Hamburg, Germany
7. United Nations University Hub on Engineering to Face Climate Change at the Hamburg University of Technology, United Nations University Institute for Water, Environment and Health (UNU-INWEH), 21073 Hamburg, Germany

<sup>*</sup> Corresponding author.

This replication package contains the code and curated data needed to reproduce the analyses, figure data, and generated supplementary tables for the article **"Land degradation is associated with larger crop yield gaps across global croplands"**.

The package is organized around two replication levels. First, the included analysis-ready data, saved model objects, causal-forest outputs, and loss outputs can be used to regenerate figure data and tables. Second, upstream data-preparation and model-fitting scripts are included for transparency, but some are computationally intensive and are not needed for routine verification of the manuscript figures and tables.

Run all scripts from the package root directory:

```r
setwd("path/to/land-degradation-yield-gaps-v4")
source("setup.R")
```

After setup checks complete, the main entry point for routine verification from the included saved outputs is:

```r
source("main.R")
```

`setup.R` checks that required R packages are available and does not run analyses. `main.R` runs the scripts in `code/00_run/` to regenerate figure-data files, plot figures from those saved figure-data files, and recreate the generated supplementary tables. It does not refit causal forests or rebuild the full analysis-ready dataset from raw raster inputs.

## Data Availability and Provenance

All input datasets used in the study are derived from publicly available sources described in the manuscript Methods and supplementary information. This package provides the curated raw inputs, intermediate files, processed analysis-ready data, saved model objects, figure-data files, and generated table outputs needed to verify and reproduce the reported results.

The package includes the derived analysis-ready aggregate yield gap, actual yield, attainable yield, and harvested area layers used in the manuscript.

Users should cite the original data providers listed in the manuscript when reusing the underlying data sources. The files in this package are provided for replication of the article.

## Statement About Rights

The authors had legitimate access to the data used in this study. The replication package contains curated public-source data products, derived analysis-ready datasets, and generated outputs used for the manuscript. 

## Package Structure

```text
code/
  00_run/        Main runner scripts for routine replication
  _shared/        Shared path, plotting, map, and helper functions
  01_data/        Data export and analysis-ready data preparation scripts
  02_models/      Regression, causal-forest, CATE, and loss-calculation scripts
  03_figures/     Figure-data and plotting scripts
    figure_data/  Scripts that create saved figure-data objects
    plotting/     Scripts that plot figures from saved figure-data objects
  04_tables/      Scripts for generated supplementary tables

data/
  raw/            Curated raw inputs included for replication
  intermediate/   Intermediate data products
  processed/      Analysis-ready data, model objects, CATE outputs, and losses

output/
  figure_data/    Source data underlying manuscript figures
  tables/         Generated supplementary tables

env/
  R_packages.txt  R package list used by the replication code
```

The manuscript figure image files are not required for replication. They can be regenerated from the code and figure-data objects.


## Software Requirements

The analyses were prepared using R 4.5.2. The package list used by the replication code is provided in:

```text
env/R_packages.txt
```

Key R packages include `terra`, `sf`, `fixest`, `grf`, `marginaleffects`, `dplyr`, `tidyr`, `readr`, `ggplot2`, `tmap`, `broom`, `classInt`, `colorspace`, `cowplot`, and `tidyterra`.

An optional `renv.lock` file is also provided for users who want to recreate the R package versions used in this environment. This package does not auto-activate `renv`; no `.Rprofile` or `renv/activate.R` file is included. To use the lockfile, install `renv` and run:

```r
install.packages("renv")
renv::restore()
```

Users may also install the packages listed in `env/R_packages.txt` manually and run the scripts without `renv`. The root-level `setup.R` script checks whether these packages are available. If missing packages should be installed automatically from CRAN, run `Sys.setenv(LDYG_INSTALL_MISSING_PACKAGES = "true")` before `source("setup.R")`.

The Google Earth Engine export script requires Python and access to Google Earth Engine. It is included for transparency but is not required to regenerate figures and tables from the data included in this package.

## Computational Requirements

Routine verification using the included analysis-ready data and saved model/output objects can be run on a standard desktop or laptop with R installed. Full upstream reconstruction is more demanding.

Approximate requirements:

- Figure-data generation and plotting from saved outputs: minutes to a few hours, depending on raster plotting settings.
- Regression model fitting: moderate to long, depending on hardware.
- Causal forest fitting: computationally intensive and unlikely to be practical on a standard laptop. In our runs, causal forest fitting was performed on a server with 16 CPU cores and 200 GB RAM; a single causal forest fit could take up to approximately 8 hours and use roughly 20-30 GB RAM, depending on the specification.
- Full-resolution raster plotting without sampling/resampling can be slow.

The replication package includes the saved causal-forest outputs used for the manuscript figures, so routine verification does not require refitting these models. The causal forest fitting scripts are included for transparency, but users who only want to reproduce the manuscript figures should use the saved CATE outputs included in `data/processed/causal_forest/`.

## Controlled Randomness

The main regression and plotting scripts are deterministic conditional on the included input files. Causal forest fitting involves random forests and should be run with the seeds/settings specified in the relevant model scripts if full re-fitting is attempted. The saved causal-forest outputs included in this package are the objects used for the manuscript figures.

## Description of Code

### Main Runners

- `main.R`: Default routine verification entry point from the package root.
- `code/00_run/01_create_figure_data.R`: Creates or refreshes figure-data objects in `output/figure_data/` from included saved outputs.
- `code/00_run/02_plot_figures.R`: Plots manuscript figures from saved figure-data objects.
- `code/00_run/03_create_tables.R`: Creates generated supplementary-table outputs.
- `code/00_run/99_full_rebuild_optional.R`: Guarded optional upstream rebuild runner for data/model/loss products.

### Data Preparation

- `code/01_data/01_export_gee_stack.py`: Google Earth Engine workflow for exporting the analysis raster stack.
- `code/01_data/02_build_stage2_grid_table.R`: Builds the grid-level table from raster inputs and spatial joins.
- `code/01_data/03_build_final_analysis_table.R`: Adds local covariates and constructs the final analysis-ready dataset.
- `code/01_data/04_create_regression_dataset.R`: Creates the regression-variable dataset used in manuscript analyses.
- `code/01_data/dataset_contract.R`: Defines expected variables for the analysis-ready data.

### Model and Loss Outputs

- `code/02_models/01_fit_regression_models.R`: Fits regression models and creates saved model objects used for coefficient plots and supplementary tables.
- `code/02_models/02_fit_causal_forests.R`: Fits causal forests. This step is computationally intensive.
- `code/02_models/03_create_cate_inspection_outputs.R`: Creates CATE outputs used for inspection and heterogeneity figures.
- `code/02_models/04_compute_loss_rasters_and_tables.R`: Computes production, calorie, protein, and revenue loss rasters and country-level summaries.

### Figure Data and Figures

For manuscript figures, replication is two-step:

1. Run the relevant script in `code/03_figures/figure_data/` to create or refresh the source-data object in `output/figure_data/`.
2. Run the corresponding script in `code/03_figures/plotting/` to plot from that saved source-data object.



### Tables
- `code/04_tables/SuppTable2_interpret_10pct_land_degradation.R`
- `code/04_tables/SuppTable3_country_losses.R`
- `code/04_tables/SuppTable5_summary_statistics.R`

These scripts create generated supplementary-table outputs in `output/tables/`.

## Instructions for Replication

1. Install R.
2. Set the working directory to the root of this package.
3. Check package requirements:

```r
source("setup.R")
```

4. For routine verification from the included saved outputs, run:

```r
source("main.R")
```

5. To regenerate source data for selected figures only, run the relevant scripts in:

```text
code/03_figures/figure_data/
```

6. To regenerate figure plots from saved figure data, run the relevant scripts in:

```text
code/03_figures/plotting/
```

7. To regenerate generated supplementary tables, run:

```text
code/04_tables/
```

8. To reproduce upstream analysis-ready data, regression models, causal forests, or loss rasters, run scripts in `code/01_data/` and `code/02_models/` in numerical order. These steps are not necessary for routine verification from the included saved outputs and may be computationally intensive.

## Suggested Execution Order

For routine figure/table verification from saved outputs:

```text
setup.R
main.R
```

Equivalent component runners are:

```text
code/00_run/01_create_figure_data.R
code/00_run/02_plot_figures.R
code/00_run/03_create_tables.R
```

For full upstream reconstruction:

```text
code/01_data/01_export_gee_stack.py
code/01_data/02_build_stage2_grid_table.R
code/01_data/03_build_final_analysis_table.R
code/01_data/04_create_regression_dataset.R
code/02_models/01_fit_regression_models.R
code/02_models/02_fit_causal_forests.R
code/02_models/03_create_cate_inspection_outputs.R
code/02_models/04_compute_loss_rasters_and_tables.R
code/03_figures/figure_data/*.R
code/03_figures/plotting/*.R
code/04_tables/*.R
```

## Notes

Some figures in the manuscript may have been manually arranged or edited for final presentation. The code and source-data files in this package provide the reproducible base outputs and underlying data used for those figures.










