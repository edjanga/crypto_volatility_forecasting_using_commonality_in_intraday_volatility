#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Model performance SAM.
#######################################################
analysis=feature_importance
python3 ../generate_extra_regression_analysis.py --analysis=$analysis
deactivate
