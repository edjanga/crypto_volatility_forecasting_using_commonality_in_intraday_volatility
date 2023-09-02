#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance SAM.
#######################################################
#regression_type=xgboost
for l in {1D,1W,1M}
  do
    python3 ./generate_results.py --L=$l --models=risk_metrics --training_scheme=SAM --regression=linear \
            --transformation=log
    python3 ./generate_results.py --L=$l --models=ar --training_scheme=SAM --regression=linear \
            --transformation=log
    for regression_type in {linear,lasso,elastic,ridge,pcr}
      do
        for model in {har,har_mkt}
          do
            python3 ./generate_results.py --L=$l --models=$model --training_scheme=SAM --regression=$regression_type \
            --transformation=log
         done
      done
  done
deactivate
