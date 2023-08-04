#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance CAM. #risk_metrics,
#######################################################
for l in {1D,1W,1M}
  do
    for regression_type in {pcr,lasso,elastic,ridge}
      do
        #for model in {ar,har,har_mkt}
          #do
            python3 ./generate_results.py --L=$l --models=$model --training_scheme=CAM \
            --regression=pcr --transformation=log &
          #done
          #wait
      done
  done
deactivate
