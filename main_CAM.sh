#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance CAM. #risk_metrics,
#######################################################
for l in {1D,1W,1M}
  do
    for regression_type in {lasso,ridge,elastic,pcr}
      do
        for model in {risk_metrics,ar,har,har_mkt}
          do
            python3 ./generate_results.py --L=$l --models=$model --training_scheme=CAM \
            --regression=$regression_type --transformation=log
          done
      done
  done
deactivate
