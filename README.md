# storm_regression

## Introduction

This repository provides the implementation of regression modelling for geomagnetic storm forecasting using solar wind ensembles. We target the Hpo indices themselves, rather than storm occurence, as done previously in Billcliff et al. (2025)

The work builds on utilities and preprocessing from [`storm_utils`](https://github.com/MBillcliff/storm_utils), and uses data generated via ambient HUXt ensemble simulations.

Billcliff et al. (2025, in review)

## Installation

1. Clone this repository and all required dependencies:

```bash
git clone https://github.com/MBillcliff/storm_utils
git clone https://github.com/University-of-Reading-Space-Science/HUXt
git clone https://github.com/mathewjowens/HUXt_tools
```

2. Create the Conda environment for this project:

```bash
cd storm_regression
conda env create -f environment.yml
conda activate storm_regression_env
```

3. Install storm_utils, HUXt, and HUXt_tools into this environment (environment must be active before running these commands):

```bash 
pip install -e ../storm_utils
pip install -e ../HUXt
pip install -e ../HUXt_tools
```

## Data requirements

To generate required data (e.g. HUXt runs, OMNI data, Hpo data), see the notebooks in [storm_utils/notebooks/](https://github.com/MBillcliff/storm_utils/notebooks/). You only need to do this once, and the output data can be shared between projects (e.g. [storm_classification](https://github.com/Mbillcliff/storm_classification/))

Shared HUXt data will be stored in the [HUXt/data/](https://github.com/University-of-Reading-Space-Science/HUXt/data) directory. 
Shared OMNI and Hpo data will be stored in the [storm_utils/data/](https://github.com/MBillcliff/storm_utils) directory.

## Contact

Please contact [Matthew Billcliff](https://github.com/MBillcliff/)