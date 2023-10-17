#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance SAM.
#######################################################
#for analysis in {coefficient_analysis,feature_importance}
#  do
#    python3 ./generate_extra_regression_analysis.py --analysis=$analysis
#  done
analysis=feature_importance
python3 ./generate_extra_regression_analysis.py --analysis=$analysis
deactivate
