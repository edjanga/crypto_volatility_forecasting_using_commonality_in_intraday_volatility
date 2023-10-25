#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Extra regression analaysis for CLustAM and CAM.
#######################################################
analysis=feature_importance
python3 ../generate_extra_regression_analysis.py --analysis=$analysis
deactivate
