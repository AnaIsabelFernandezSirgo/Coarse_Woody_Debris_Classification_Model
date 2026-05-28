# Coarse Woody Debris Classification Model

## Overview

This repository contains the datasets and machine learning scripts used for coarse woody debris (CWD) sediment-storage classification on burned hillslopes using field measurements collected in the southwestern United States.

## Data

Field measurements were collected from seven 25 m × 25 m plots located within two wildfire perimeters in the southwestern United States. Two plots were located within the 2010 Schultz Fire in Arizona (plot IDs 1–2), and five plots were located within the 2018 Buzzard Fire in New Mexico (plot IDs 3–7).

### Input File

- `CWD_dataset.xlsx` — field measurements, sediment-storage measurements, and terrain attributes used for model development and evaluation.

## Scripts

- `All_Models.py` — builds and evaluates models using all possible combinations of the selected features for 3-, 4-, 5-, 6-, and 7-feature models.
- `Model_Train.py` — trains the final selected model.
- `MODEL_CWD.py` — implementation of the final selected coarse woody debris classification model.
- `Feature_importance.py` — permutation feature importance analysis.
- `PDP_SH.py` — generates partial dependence plots (PDP) and SHAP visualizations.
