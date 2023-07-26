#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Model performance SAM.
#######################################################
l=1D
model=har
regression_type=linear
#for l in {1D,1W,1M}
  #do
    #python3 ./generate_results.py --L=$l --models=risk_metrics --training_scheme=SAM --regression=linear \
    #        --transformation=log &
    #wait
    #for regression_type in {linear,lasso,elastic,ridge}
      #do
        #for model in {ar,har,har_dummy_markets}
          #do
            python3 ./generate_results.py --L=$l --models=$model --training_scheme=SAM --regression=$regression_type \
            --transformation=log
         #done
         #wait
      #done
  #done
deactivate
