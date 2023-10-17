#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance ClustAM.
#######################################################
l=1W
#for l in {6M,1M,1W}
#  do
    #for regression_type in {lightgbm,elastic,lasso,pcr}
      #do
        for model in {har_eq,har,ar}
          do
            python3 ./generate/generate_results.py --L=$l --model=$model --training_scheme=ClustAM \
            --regression=pcr --transformation=log
          done
      #done
#  done
deactivate
