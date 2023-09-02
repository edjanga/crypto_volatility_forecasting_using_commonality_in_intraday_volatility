#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance UAM. #risk_metrics,
#######################################################
for l in {1W,1M}
  do
    for regression_type in {lstm,lstm_transformers}
      do
        for model in {risk_metrics,ar,har,har_mkt}
          do
            python3 ./generate_results.py --L=$l --models=$model --training_scheme=UAM \
            --regression=$regression_type --transformation=log
          done
      done
  done
deactivate
