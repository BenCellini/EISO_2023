# EISO_2023

This repository contains the core code (see 'code' folder) for constructing an empirical observability matrix and determining the approximate observability of induvidual state variables. All code for making data & figures corresponding to the preprint can be found in the 'notebooks' folder as jupyter notebooks.

### Overview of where to find main functions, classes, and data:

* Constructing observability matrix: <code>observability.py</code>
* Induvidual state observability from observability matrix: <code>eiso.py</code>
* Simulating linear system: <code>simulator.py</code>
* Simulating fly-wind system: <code>fly_wind_simple.py</code>
* Functions for making pretty fly trajectory figures: <code>figure_functions.py</code>
* Analytical observability analysis for fly-wind system: <code>fly_wind_analytical.ipynb</code>
* Data generated from fly-wind system for Figure 3: <code>simulation_data</code> folder
* Vector format figures: <code>figures</code> folder