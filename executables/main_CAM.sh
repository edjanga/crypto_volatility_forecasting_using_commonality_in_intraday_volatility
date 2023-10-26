#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Model performance CAM.
#######################################################
for l in {1W,1M,6M}
  do
    for regression_type in {lightgbm,elastic,lasso,pcr}
      do
        for model in {har_eq,har,ar}
          do
            python3 ../generate_results.py --L=$l --model=$model --training_scheme=CAM --regression=$regression_type \
            --transformation=log
          done
      done
  done

