#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance CAM.
#######################################################
for l in {1D,1W,1M}
  do
    for regression_type in {lasso,elastic,ridge}
      do
        for model in {ar,risk_metrics,har,har_mkt}
          do
            python3 ./generate_results.py --L=$l --models=$model --training_scheme=CAM \
            --regression=$regression_type --transformation=log &
          done
          wait
      done
  done
deactivate
