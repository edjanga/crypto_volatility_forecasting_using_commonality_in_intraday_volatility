#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance CAM.
#######################################################
l=1W
#for l in {6M,1M,1W} #lightgbm,elastic,
#  do
    for regression_type in {lasso,pcr}
      do
        #for model in {har_eq,har,ar}
          #do
            python3 ./generate_results.py --L=$l --model=ar --training_scheme=CAM --regression=lasso \
            --transformation=log
          #done
      done
#  done
deactivate
