#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance ClustAM.
#######################################################
regression_type=pcr
for l in {1D,1W,1M}
  do
    #for regression_type in {lasso,ridge,elastic}
      #do
        for model in {ar,har,har_mkt}
          do
            python3 ./generate_results.py --L=$l --models=ar --training_scheme=ClustAM \
            --regression=$regression_type --transformation=log
          done
          #wait
      #done
      #wait
  done
deactivate
