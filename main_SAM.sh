#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance SAM.
#######################################################
for l in {1D,1W,1M}
  do
    python3 ./generate_results.py --L=$l --models=$model --training_scheme=SAM --regression=risk_metrics \
            --transformation=log
    for regression_type in {linear,lasso,elastic,ridge}
      do
        for model in {ar,har,har_mkt}
          do
            python3 ./generate_results.py --L=$l --models=$model --training_scheme=SAM --regression=$regression_type \
            --transformation=log
         done
         wait
      done
  done
deactivate
