#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance SAM.
#######################################################
for l in {6M,1M,1W}
  do
#    python3 ./generate_results.py --L=$l --model=risk_metrics --training_scheme=SAM --regression=linear \
#            --transformation=log
    for regression_type in {lightgbm,elastic,lasso,pcr}
      do
        for model in {har_eq,har}
          do
            python3 ./generate_results.py --L=$l --model=$model --training_scheme=SAM \
            --regression=$regression_type --transformation=log &
          done
          wait
      done
    python3 ./generate_results.py --L=$l --model=ar --training_scheme=SAM --regression=linear \
    --transformation=log
  done
deactivate
