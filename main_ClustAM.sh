#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance ClustAM.
#######################################################
for l in {1D,1M,1W}
  do
    for regression_type in {lasso,ridge,elastic}
      do
        for model in {ar,risk_metrics,har,har_dummy_markets}
          do
            python3 ./generate_results.py --L=$l --models=$model --training_scheme=ClustAM \
            --regression=$regression_type --transformation=log &
          done
        wait
      done
  done
deactivate
