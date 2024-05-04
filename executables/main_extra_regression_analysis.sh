#!/bin/bash
clear
source ../venv/bin/activate
#######################################################################
## Extra regression analysis for CLustAM and CAM + Spillover effect.
#######################################################################
# analysis=feature_importance
# python3 ../generate_extra_regression_analysis.py --analysis=$analysis
python3 ../generate_spillover.py
deactivate
