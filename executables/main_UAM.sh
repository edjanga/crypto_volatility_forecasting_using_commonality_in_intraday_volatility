#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance UAM.
#######################################################
for l in {1W,1M,6M}
  do
  if [ "$l" != "6M" ]; then
    regression_type_list=("lasso" "elastic" "pcr" "lightgbm")
  else
    regression_type_list=("lasso" "elastic" "pcr" "lightgbm" "lstm")
  fi
  for regression_type in $regression_type_list
    do
      for model in {ar,har,har_eq}
        do
          python3 ./generate_results.py --L=$l --model=$model --training_scheme=UAM \
          --regression=$regression_type --transformation=log
        done
    done
  done
deactivate
